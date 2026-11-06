"""RoboHeard: 试听比较六通道 AEC 音频，选取参考麦克风。

用途：截取同一时间段，导出 6 个独立单通道 WAV，附带基础音量与削波统计。
注意：统计数据不能自动判定最佳通道；需要听不同说话人是否被保留。

示例：
    python scripts/compare_channels.py --input data/raw/Audio_AEC/S150.wav --start 40 --seconds 15
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def dbfs(value: float) -> str:
    """把线性振幅变成 dBFS（数字音频相对于满刻度的分贝）。"""
    if value <= 0:
        return "-inf"
    return f"{20 * math.log10(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="导出六通道 AEC 对照试听音频")
    parser.add_argument("--input", type=Path, required=True, help="六通道 WAV 路径")
    parser.add_argument("--start", type=float, default=40.0, help="截取起始时间（秒）")
    parser.add_argument("--seconds", type=float, default=15.0, help="截取持续时间（秒）")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/channel_candidates"),
        help="输出目录",
    )
    args = parser.parse_args()

    if args.start < 0 or args.seconds <= 0:
        parser.error("--start 不能为负；--seconds 必须大于 0")
    if not args.input.is_file():
        parser.error(f"未找到音频文件：{args.input}")

    with sf.SoundFile(args.input) as wav:
        if wav.channels != 6:
            parser.error(f"预期六通道音频，实际通道数为 {wav.channels}")
        if wav.samplerate != 16_000:
            parser.error(f"预期采样率 16000 Hz，实际为 {wav.samplerate} Hz")
        start_sample = round(args.start * wav.samplerate)
        if start_sample >= len(wav):
            parser.error("起始时间超出音频时长")
        frame_count = min(round(args.seconds * wav.samplerate), len(wav) - start_sample)
        wav.seek(start_sample)
        # always_2d=True: 保持形状 [Samples, Channels]
        audio = wav.read(frames=frame_count, dtype="float32", always_2d=True)
        rate = wav.samplerate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_str = f"{args.start:g}".replace(".", "p")
    dur_str = f"{len(audio) / rate:g}".replace(".", "p")
    tag = f"{args.input.stem}_s{start_str}_d{dur_str}"
    rows = []

    for index in range(6):
        wave = audio[:, index]
        rms = float(np.sqrt(np.mean(np.square(wave.astype(np.float64)))))
        peak = float(np.max(np.abs(wave)))
        clip_pct = 100 * float(np.mean(np.abs(wave) >= 0.999))
        file_path = args.output_dir / f"{tag}_mic{index + 1}.wav"
        # 保留原幅度，不逐通道归一化，以免掩盖真实音量差异。
        sf.write(file_path, wave, rate, subtype="PCM_16")
        rows.append({
            "channel": index + 1,
            "file": file_path.name,
            "rms_dbfs": dbfs(rms),
            "peak_dbfs": dbfs(peak),
            "clip_pct": f"{clip_pct:.4f}",
        })

    csv_path = args.output_dir / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "file", "rms_dbfs", "peak_dbfs", "clip_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"输入：{args.input}")
    print(f"时段：{args.start:g} s 至 {args.start + len(audio) / rate:g} s")
    print(f"采样率：{rate} Hz，通道数：6")
    print(f"导出目录：{args.output_dir.resolve()}")
    print("\n通道 | RMS(dBFS) | 峰值(dBFS) | 削波比例")
    for r in rows:
        print(f"Mic {r['channel']} | {r['rms_dbfs']:>9} | {r['peak_dbfs']:>12} | {r['clip_pct']:>8}%")
    print(f"\n统计表：{csv_path.name}")
    print("注意：音量最大不等于最佳通道；请试听是否保留了所有说话人的声音。")


if __name__ == "__main__":
    main()
