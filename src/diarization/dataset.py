
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class RoboHeardDataset(Dataset):
    """
    RoboHeard 音频及帧级说话人标签数据集。

    audio:  [128000]
    labels: [399, 7]

    训练时随机选择一个麦克风；
    验证时固定使用第 0 号通道。
    """

    def __init__(
        self,
        audio_dir,
        mode="train",
        chunk_duration=8.0,
        stride=4.0,
        sample_rate=16000,
        annotation_dir=None,
        num_speakers=7,
        session_ids=None,
    ):
        if mode not in ("train", "val"):
            raise ValueError("mode 必须是 train 或 val")

        if sample_rate != 16000:
            raise ValueError("当前标签网格要求 16 kHz 采样率")

        self.audio_dir = Path(audio_dir)
        self.mode = mode
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers

        self.chunk_samples = int(
            chunk_duration * sample_rate
        )
        self.stride_samples = int(
            stride * sample_rate
        )

        self.annotation_dir = (
            Path(annotation_dir)
            if annotation_dir is not None
            else None
        )

        if self.chunk_samples < 400:
            raise ValueError("音频片段太短")

        if self.stride_samples <= 0:
            raise ValueError("stride 必须大于 0")

        # WavLM-Large 卷积前端：
        # 总步长 320 samples，感受野 400 samples
        self.frame_hop = 320
        self.frame_receptive = 400

        # 8 秒音频对应 399 个特征帧
        self.num_frames = (
            1
            + (self.chunk_samples - self.frame_receptive)
            // self.frame_hop
        )

        # 每个输出帧在当前音频片段中的中心时间
        self.frame_centers = (
            np.arange(
                self.num_frames,
                dtype=np.float64,
            ) * self.frame_hop
            + (self.frame_receptive - 1) / 2
        ) / self.sample_rate

        # 保存音频片段索引
        self.samples = []

        # 保存每个 Session 的 JSON 标注
        self.segments_by_session = {}

        # 保存每个 Session 的说话人槽位映射
        self.speaker_to_slot = {}

        allowed = (
            set(session_ids)
            if session_ids is not None
            else None
        )

        audio_files = sorted(
            self.audio_dir.glob("*.wav")
        )

        file_count = 0

        for audio_path in audio_files:

            session_id = audio_path.stem

            if (
                allowed is not None
                and session_id not in allowed
            ):
                continue

            info = sf.info(str(audio_path))

            if (
                info.samplerate != sample_rate
                or info.channels != 6
            ):
                raise ValueError(
                    f"音频格式不符合要求: {audio_path}"
                )

            file_count += 1

            # 如果提供了 JSON 目录，则加载监督标签
            if self.annotation_dir is not None:

                json_path = (
                    self.annotation_dir
                    / f"{session_id}.json"
                )

                if not json_path.is_file():
                    raise FileNotFoundError(
                        f"缺少对应 JSON: {json_path}"
                    )

                with json_path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    annotation = json.load(f)

                if annotation["session_id"] != session_id:
                    raise ValueError(
                        f"Session ID 不匹配: {json_path}"
                    )

                segments = annotation["segments"]

                # 当前 Session 中所有不同的说话人
                speaker_ids = sorted({
                    seg["speaker_id"]
                    for seg in segments
                })

                if len(speaker_ids) > num_speakers:
                    raise ValueError(
                        f"{session_id} 超过 "
                        f"{num_speakers} 个说话人"
                    )

                # 例如 P054 -> 0, P055 -> 1
                speaker_mapping = {
                    speaker_id: i
                    for i, speaker_id
                    in enumerate(speaker_ids)
                }

                self.segments_by_session[
                    session_id
                ] = segments

                self.speaker_to_slot[
                    session_id
                ] = speaker_mapping

            # 按 8 秒长度、4 秒步长建立索引
            if info.frames < self.chunk_samples:
                starts = [0]
            else:
                starts = range(
                    0,
                    info.frames - self.chunk_samples + 1,
                    self.stride_samples,
                )

            for start_sample in starts:
                self.samples.append(
                    (audio_path, start_sample)
                )

        print("模式:", self.mode)
        print("音频文件数量:", file_count)
        print("音频片段数量:", len(self.samples))
        print("标签帧数:", self.num_frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        audio_path, start_sample = self.samples[index]
        session_id = audio_path.stem

        # 只读取当前 8 秒，不加载整个 WAV
        with sf.SoundFile(str(audio_path)) as file:

            file.seek(start_sample)

            audio = file.read(
                frames=self.chunk_samples,
                dtype="float32",
                always_2d=True,
            )

        valid_samples = len(audio)

        # 不足 8 秒时补零
        if valid_samples < self.chunk_samples:

            audio = np.pad(
                audio,
                (
                    (
                        0,
                        self.chunk_samples - valid_samples,
                    ),
                    (0, 0),
                ),
            )

        # 训练随机选择麦克风，验证固定使用 Mic 1
        if self.mode == "train":
            channel = random.randrange(6)
        else:
            channel = 0

        waveform = torch.from_numpy(
            np.ascontiguousarray(
                audio[:, channel]
            )
        )

        start_time = start_sample / self.sample_rate

        result = {
            "audio": waveform,
            "session_id": session_id,
            "start_time": start_time,
            "channel": channel,
        }

        # 如果没有提供 JSON，保持原来的纯音频行为
        if self.annotation_dir is None:
            return result

        # -------------------------------------
        # 生成当前片段对应的帧级标签
        # -------------------------------------

        labels = np.zeros(
            (self.num_frames, self.num_speakers),
            dtype=np.float32,
        )

        # 将相对帧时间转换为整场 Session 的绝对时间
        frame_times = start_time + self.frame_centers

        window_end = (
            start_time
            + valid_samples / self.sample_rate
        )

        segments = self.segments_by_session[
            session_id
        ]

        speaker_mapping = self.speaker_to_slot[
            session_id
        ]

        for seg in segments:

            seg_start = seg["start"]
            seg_end = seg["end"]

            # 跳过与当前音频片段没有交集的标注
            if (
                seg_end <= start_time
                or seg_start >= window_end
            ):
                continue

            # 当前说话人的槽位编号
            speaker_slot = speaker_mapping[
                seg["speaker_id"]
            ]

            # 判断哪些时间帧位于讲话区间内
            active = (
                (frame_times >= seg_start)
                & (frame_times < seg_end)
            )

            # 对应帧标记为 1
            # 不会清除其他说话人的活动标签
            labels[active, speaker_slot] = 1.0

        # 记录真实音频对应的有效帧数
        # 供后面 Loss 屏蔽补零区域
        valid_frames = max(
            0,
            1 + (
                valid_samples - self.frame_receptive
            ) // self.frame_hop,
        )

        result["labels"] = torch.from_numpy(
            labels
        )

        result["valid_frames"] = valid_frames

        return result


if __name__ == "__main__":

    dataset = RoboHeardDataset(
        audio_dir="data/raw/Audio_AEC",
        annotation_dir="data/raw/DOC_AEC_JSON",
        mode="val",
        chunk_duration=8.0,
        stride=4.0,
        num_speakers=7,
    )

    sample = dataset[0]

    print("Session:", sample["session_id"])

    print(
        "槽位对应关系:",
        dataset.speaker_to_slot[
            sample["session_id"]
        ],
    )

    print(
        "音频形状:",
        tuple(sample["audio"].shape),
    )

    print(
        "标签形状:",
        tuple(sample["labels"].shape),
    )

    print(
        "有效标签帧数:",
        sample["valid_frames"],
    )

    print(
        "最多同时活动人数:",
        int(
            sample["labels"]
            .sum(dim=-1)
            .max()
            .item()
        ),
    )

    assert sample["audio"].shape == (128000,)
    assert sample["labels"].shape == (399, 7)
    assert sample["valid_frames"] == 399

    print("音频与帧级标签测试通过！")
