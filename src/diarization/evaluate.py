"""RoboHeard Hyper-SD: 8s window DER and approximate session-level DER.

Run from the RoboHeard project root:
    python -m src.diarization.evaluate --session S150 --split all
    python -m src.diarization.evaluate --split val

This is a 20-ms, frame-discretized, zero-collar diagnostic metric. It is
not a replacement for the competition's official diarization scorer.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from .model import HyperSDModel


SAMPLE_RATE = 16000
CHANNELS = 6
CHUNK_SAMPLES = 8 * SAMPLE_RATE
STRIDE_SAMPLES = 4 * SAMPLE_RATE
FRAME_HOP = 320
FRAME_RECEPTIVE = 400
NUM_SPEAKERS = 7


def frame_count(num_samples):
    """Number of WavLM output frames whose receptive fields are fully valid."""
    if num_samples < FRAME_RECEPTIVE:
        return 0
    return 1 + (num_samples - FRAME_RECEPTIVE) // FRAME_HOP


def chunk_starts(num_samples):
    """4s-aligned chunks, with a padded final chunk only if needed."""
    if num_samples < FRAME_RECEPTIVE:
        raise ValueError("音频短于 WavLM 卷积前端感受野")
    if num_samples <= CHUNK_SAMPLES:
        return [0]
    starts = list(range(0, num_samples - CHUNK_SAMPLES + 1, STRIDE_SAMPLES))
    if starts[-1] + CHUNK_SAMPLES < num_samples:
        starts.append(starts[-1] + STRIDE_SAMPLES)
    return starts


def load_reference(json_path, num_samples, max_speakers=NUM_SPEAKERS):
    """Convert JSON intervals to binary reference labels on WavLM's time grid."""
    with Path(json_path).open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    segments = annotation["segments"]
    speaker_ids = sorted({seg["speaker_id"] for seg in segments})
    if len(speaker_ids) > max_speakers:
        raise ValueError(f"{json_path}: 检测到 {len(speaker_ids)} 人，超过 {max_speakers}")

    total_frames = frame_count(num_samples)
    center_times = (
        np.arange(total_frames, dtype=np.float64) * FRAME_HOP
        + (FRAME_RECEPTIVE - 1) / 2
    ) / SAMPLE_RATE

    reference = np.zeros((total_frames, len(speaker_ids)), dtype=np.bool_)
    speaker_to_col = {speaker: i for i, speaker in enumerate(speaker_ids)}

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        left = np.searchsorted(center_times, start, side="left")
        right = np.searchsorted(center_times, end, side="left")
        reference[left:right, speaker_to_col[segment["speaker_id"]]] = True

    return reference, speaker_ids


def score_der(reference, hypothesis):
    """Overlap-aware discrete DER with one optimal mapping over ALL supplied frames.

    reference: [T, R] binary, hypothesis: [T, H] binary.
    Mapping is optimized on speaker co-occurrence using Hungarian assignment.
    """
    ref = np.asarray(reference, dtype=np.bool_)
    hyp = np.asarray(hypothesis, dtype=np.bool_)
    if ref.ndim != 2 or hyp.ndim != 2 or ref.shape[0] != hyp.shape[0]:
        raise ValueError("参考标签和预测必须是时间帧数相同的二维矩阵")

    num_frames = ref.shape[0]
    correct_by_frame = np.zeros(num_frames, dtype=np.int64)

    if ref.shape[1] > 0 and hyp.shape[1] > 0:
        overlaps = (
            ref.astype(np.int64).T @ hyp.astype(np.int64)
        )
        rows, cols = linear_sum_assignment(overlaps, maximize=True)
        correct_by_frame = np.logical_and(ref[:, rows], hyp[:, cols]).sum(axis=1)

    ref_count = ref.sum(axis=1).astype(np.int64)
    hyp_count = hyp.sum(axis=1).astype(np.int64)

    miss = int(np.maximum(ref_count - hyp_count, 0).sum())
    false_alarm = int(np.maximum(hyp_count - ref_count, 0).sum())
    confusion = int((np.minimum(ref_count, hyp_count) - correct_by_frame).sum())
    total = int(ref_count.sum())

    return {"miss": miss, "false_alarm": false_alarm,
            "confusion": confusion, "reference": total}


def add_scores(total, partial):
    for key in total:
        total[key] += partial[key]


def make_score():
    return {"miss": 0, "false_alarm": 0, "confusion": 0, "reference": 0}


def render_scores(score):
    """Convert 20-ms speaker-frame counts to seconds and DER percent."""
    ref = score["reference"]
    errors = score["miss"] + score["false_alarm"] + score["confusion"]
    return {
        "der_percent": None if ref == 0 else round(100.0 * errors / ref, 3),
        "miss_seconds": round(score["miss"] * FRAME_HOP / SAMPLE_RATE, 3),
        "false_alarm_seconds": round(score["false_alarm"] * FRAME_HOP / SAMPLE_RATE, 3),
        "confusion_seconds": round(score["confusion"] * FRAME_HOP / SAMPLE_RATE, 3),
        "reference_speaker_seconds": round(ref * FRAME_HOP / SAMPLE_RATE, 3),
    }


class OverlapTrackStitcher:
    """Join window-local speaker slots using speech overlap in adjacent windows.

    This uses prediction only, never JSON speaker identities. If the overlap
    contains insufficient evidence, the local speaker becomes a new global
    track. Fragmentation can increase session DER; this is intentional.
    """

    def __init__(self, total_frames, threshold=0.5,
                 min_iou=0.10, min_intersection=3, min_active_frames=2):
        self.total_frames = total_frames
        self.threshold = threshold
        self.min_iou = min_iou
        self.min_intersection = min_intersection
        self.min_active_frames = min_active_frames
        self.tracks_sum = []
        self.tracks_count = []
        self.coverage = np.zeros(total_frames, dtype=np.int16)

    def add_window(self, start_sample, probabilities, valid_frames):
        """probabilities [399,7]; all chunk starts are multiples of 4 s."""
        if start_sample % FRAME_HOP:
            raise ValueError("窗口起点必须落在 20ms 帧网格上")
        start_frame = start_sample // FRAME_HOP
        positions = start_frame + np.arange(valid_frames)
        if positions.size == 0 or positions[-1] >= self.total_frames:
            raise ValueError("窗口时间位置超出会话范围")

        probs = np.asarray(probabilities[:valid_frames], dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != NUM_SPEAKERS:
            raise ValueError(f"需要 [T,{NUM_SPEAKERS}] 的说话人概率")

        binary = probs >= self.threshold
        active_locals = np.flatnonzero(
            np.sum(binary, axis=0) >= self.min_active_frames
        )
        assignments = {}  # local slot -> global track index

        # Compare only frames already predicted by previous windows.
        overlap = self.coverage[positions] > 0
        if self.tracks_sum and active_locals.size and np.any(overlap):
            overlap_positions = positions[overlap]
            previous = np.stack([
                np.divide(
                    s[overlap_positions],
                    np.maximum(c[overlap_positions], 1),
                )
                for s, c in zip(self.tracks_sum, self.tracks_count)
            ], axis=1)

            old_active = previous >= self.threshold
            new_active = binary[overlap][:, active_locals]
            intersections = (
                old_active.astype(np.int64).T
                @ new_active.astype(np.int64)
            )
            old_counts = old_active.sum(axis=0)
            new_counts = new_active.sum(axis=0)
            unions = old_counts[:, None] + new_counts[None, :] - intersections
            ious = np.divide(
                intersections, np.maximum(unions, 1), dtype=np.float64
            )

            global_rows, local_cols = linear_sum_assignment(ious, maximize=True)
            for g, local_pos in zip(global_rows, local_cols):
                if (intersections[g, local_pos] >= self.min_intersection
                        and ious[g, local_pos] >= self.min_iou):
                    assignments[int(active_locals[local_pos])] = int(g)

        # Do not silently reuse identities when overlap offers no evidence.
        for local in active_locals:
            local = int(local)
            if local not in assignments:
                assignments[local] = len(self.tracks_sum)
                self.tracks_sum.append(np.zeros(self.total_frames, dtype=np.float32))
                self.tracks_count.append(np.zeros(self.total_frames, dtype=np.int16))

        for local, global_id in assignments.items():
            self.tracks_sum[global_id][positions] += probs[:, local]
            self.tracks_count[global_id][positions] += 1

        self.coverage[positions] += 1

    def get_binary(self):
        if not self.tracks_sum:
            return np.zeros((self.total_frames, 0), dtype=np.bool_)
        columns = [
            (s / np.maximum(c, 1)) >= self.threshold
            for s, c in zip(self.tracks_sum, self.tracks_count)
        ]
        return np.stack(columns, axis=1)


def choose_session_ids(audio_dir, annotation_dir, split, seed, val_ratio,
                       only_session=None, max_sessions=0):
    ids = sorted(p.stem for p in Path(audio_dir).glob("*.wav"))
    if not ids:
        raise FileNotFoundError(f"音频目录中没有 WAV: {audio_dir}")
    if only_session:
        if only_session not in ids:
            raise FileNotFoundError(f"找不到 {only_session}.wav")
        selected = [only_session]
    else:
        shuffled = ids.copy()
        random.Random(seed).shuffle(shuffled)
        if len(ids) < 2 or val_ratio == 0:
            train_ids, val_ids = shuffled, []
        else:
            val_count = max(1, round(len(ids) * val_ratio))
            val_count = min(val_count, len(ids) - 1)
            val_ids, train_ids = shuffled[:val_count], shuffled[val_count:]
        selected = {
            "all": ids,
            "train": train_ids,
            "val": val_ids,
        }[split]
        if not selected:
            raise ValueError("没有验证 Session；单文件测试请用 --split all 或 --session S150")
    if max_sessions > 0:
        selected = selected[:max_sessions]
    for session_id in selected:
        if not (Path(annotation_dir) / f"{session_id}.json").is_file():
            raise FileNotFoundError(f"缺少标注: {session_id}.json")
    return selected


def load_checkpoint(model, checkpoint_path):
    """Restore trainable parameters, plus non-WavLM buffers if present."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = checkpoint.get("model_state", checkpoint.get("trainable_state"))
    if saved is None:
        raise KeyError("checkpoint 中没有 model_state 或 trainable_state")
    current = model.state_dict()
    missing = [
        name for name, param in model.named_parameters()
        if param.requires_grad and name not in saved
    ]
    if missing:
        raise ValueError(f"checkpoint 缺少可训练参数: {missing[:5]}")
    for name, value in saved.items():
        if name not in current:
            raise ValueError(f"checkpoint 存在未知参数: {name}")
        if current[name].shape != value.shape:
            raise ValueError(f"checkpoint 参数维度不一致: {name}")
    model.load_state_dict(saved, strict=False)
    if "model_state" not in checkpoint:
        print("注意：旧 checkpoint 未保存 Conformer BatchNorm 的运行统计，"
              "此 DER 仅适合排查流程。正式训练前请升级保存函数。")
    return checkpoint


def evaluate_session(model, device, session_id, audio_dir, annotation_dir,
                     threshold, amp, min_iou, min_intersection,
                     min_active_frames, log_every):
    wav_path = Path(audio_dir) / f"{session_id}.wav"
    json_path = Path(annotation_dir) / f"{session_id}.json"
    info = sf.info(str(wav_path))
    if info.samplerate != SAMPLE_RATE or info.channels != CHANNELS:
        raise ValueError(f"{session_id}: 要求 16kHz、6 通道 WAV")
    total_frames = frame_count(info.frames)
    ref, speakers = load_reference(json_path, info.frames)
    stitcher = OverlapTrackStitcher(
        total_frames, threshold=threshold, min_iou=min_iou,
        min_intersection=min_intersection, min_active_frames=min_active_frames,
    )
    windows_score = make_score()
    starts = chunk_starts(info.frames)

    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)
    with sf.SoundFile(str(wav_path)) as wav_file, torch.inference_mode():
        for window_i, start in enumerate(starts, start=1):
            wav_file.seek(start)
            raw = wav_file.read(
                frames=CHUNK_SAMPLES, dtype="float32", always_2d=True
            )
            valid_samples = raw.shape[0]
            valid_frames = frame_count(valid_samples)
            if valid_frames == 0:
                continue
            waveform = torch.from_numpy(np.ascontiguousarray(raw[:, 0]))[None]
            waveform = waveform.to(device)
            valid_tensor = torch.tensor([valid_frames], dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp and device.type == "cuda"):
                output = model(waveform, valid_frames=valid_tensor)
            probs = output["speaker_probs"][0].float().cpu().numpy()
            if probs.shape != (valid_frames, NUM_SPEAKERS):
                raise RuntimeError(f"模型输出形状不正确: {probs.shape}")
            prediction = probs[:valid_frames] >= threshold

            # WINDOW diagnostic: each window gets its own oracle speaker mapping.
            first = start // FRAME_HOP
            current_ref = ref[first:first + valid_frames]
            add_scores(windows_score, score_der(current_ref, prediction))

            # SESSION metric: assign slots using predictions/overlap only.
            stitcher.add_window(start, probs, valid_frames)

            if log_every and (window_i % log_every == 0 or window_i == len(starts)):
                print(f"  {session_id}: {window_i}/{len(starts)} 个窗口", flush=True)

    session_prediction = stitcher.get_binary()
    if not np.all(stitcher.coverage > 0):
        raise RuntimeError(f"{session_id}: 部分时间帧没有被任何窗口覆盖")
    session_score = score_der(ref, session_prediction)
    return {
        "session_id": session_id,
        "reference_speakers": speakers,
        "predicted_tracks": int(session_prediction.shape[1]),
        "windows": len(starts),
        "window_counts": windows_score,
        "session_counts": session_score,
        "window_der": render_scores(windows_score),
        "session_der": render_scores(session_score),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RoboHeard DER 评估")
    parser.add_argument("--audio-dir", default="data/raw/Audio_AEC")
    parser.add_argument("--annotation-dir", default="data/raw/DOC_AEC_JSON")
    parser.add_argument("--checkpoint", default=None,
                        help="默认优先 best.pt，否则 last.pt")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--session", default=None, help="例如 S150；覆盖 --split")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.10)
    parser.add_argument("--min-intersection", type=int, default=3,
                        help="跨窗口匹配所需重叠的活动帧数")
    parser.add_argument("--min-active-frames", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", default="results/diarization/eval.json")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("threshold 必须在 (0,1) 之间")
    if not 0 <= args.val_ratio < 1:
        parser.error("val-ratio 必须在 [0,1) 之间")
    if not 0 <= args.min_iou <= 1:
        parser.error("min-iou 必须在 [0,1] 之间")
    if args.min_intersection < 1 or args.min_active_frames < 1:
        parser.error("min-intersection/min-active-frames 必须 ≥1")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        Path("checkpoints/diarization/best.pt")
        if Path("checkpoints/diarization/best.pt").is_file()
        else Path("checkpoints/diarization/last.pt")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    session_ids = choose_session_ids(
        args.audio_dir, args.annotation_dir, args.split,
        args.seed, args.val_ratio, args.session, args.max_sessions,
    )
    print(f"设备: {device}; checkpoint: {checkpoint_path}")
    print(f"评估范围: {len(session_ids)} 个 Session; split={args.split}")
    if args.split == "all" and not args.session:
        print("注意：all 包含训练数据；正式验证请使用 --split val")

    model = HyperSDModel(
        num_speakers=NUM_SPEAKERS, hidden_dim=256,
        embedding_dim=128, conformer_layers=4,
    ).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path)
    print("加载 checkpoint epoch:", checkpoint.get("epoch", "未知"))
    model.eval()

    overall_window = make_score()
    overall_session = make_score()
    details = []
    for session_id in session_ids:
        result = evaluate_session(
            model, device, session_id, args.audio_dir, args.annotation_dir,
            args.threshold, not args.no_amp, args.min_iou,
            args.min_intersection, args.min_active_frames, args.log_every,
        )
        details.append(result)
        add_scores(overall_window, result["window_counts"])
        add_scores(overall_session, result["session_counts"])
        print(
            f"{session_id}: windows={result['windows']}, "
            f"真实人数={len(result['reference_speakers'])}, "
            f"拼接轨道数={result['predicted_tracks']}, "
            f"窗口DER={result['window_der']['der_percent']}%, "
            f"Session DER={result['session_der']['der_percent']}%"
        )

    window_der = render_scores(overall_window)
    session_der = render_scores(overall_session)
    print("\n========== DER 汇总 ==========")
    print("窗口级 DER（窗口重复计入）:", window_der)
    print("Session 级 DER（每帧只计一次）:", session_der)
    print("注意：无 collar，保留重叠；窗口拼接基于重叠语音匹配，非官方评分。")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "scope": "explicit session" if args.session else args.split,
        "sessions": session_ids,
        "threshold": args.threshold,
        "frame_hop_seconds": FRAME_HOP / SAMPLE_RATE,
        "collar_seconds": 0.0,
        "overlap_scored": True,
        "window_metric": window_der,
        "session_metric_approximate": session_der,
        "details": details,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"详细结果保存至: {out_path}")


if __name__ == "__main__":
    main()
