"""RoboHeard: test six-channel WPE dereverberation on a short AEC clip.

Usage (from the project root, in the roboheard Conda environment):
    python scripts/test_wpe.py --input data/raw/Audio_AEC/S150.wav --start 40 --seconds 10

Dependencies: numpy, scipy, soundfile, nara_wpe
"""

# argparse：读取终端传入的 --input、--start 等选项
# Path：跨平台地处理文件路径，避免手动拼接路径字符串
import argparse
from pathlib import Path

# numpy 负责数组运算；soundfile 负责 WAV 读写
# STFT 把时域波形转为时频谱，ISTFT 再把频谱恢复成波形
# wpe_v8：NARA-WPE 提供的去混响算法，我们直接调用，不必自己推导
import numpy as np
import soundfile as sf
from scipy.signal import istft, stft
from nara_wpe.wpe import wpe_v8


def multi_stft(wave: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """Input [samples, channels]; output [freq, channels, frames]."""
    # wave 是一个二维数组：[采样点数, 麦克风数]；此处麦克风数为 6
    spectra = []
    for ch in range(wave.shape[1]):  # 依次处理第 1 到第 6 个麦克风
        _, _, z = stft(
            wave[:, ch],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        # z 是当前通道的复数频谱：[频率格数, 时间帧数]
        spectra.append(z)
    # 把六个通道合并成 [频率格数, 6, 时间帧数]，匹配 wpe_v8 的输入
    return np.stack(spectra, axis=1)


def multi_istft(spec: np.ndarray, sr: int, n_fft: int, hop: int, target_len: int) -> np.ndarray:
    """Input [freq, channels, frames]; output [samples, channels]."""
    # WPE 后的 spec 仍是 [频率格数, 6, 时间帧数]
    channels = []
    for ch in range(spec.shape[1]):  # 将六路频谱分别还原为音频
        _, x = istft(
            spec[:, ch, :],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        # 逆变换可能产生多/少几个采样点，统一修正到原片段长度
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)))
        channels.append(x[:target_len])
    # 输出重新变为 [采样点数, 6]
    return np.stack(channels, axis=1)


def main() -> None:
    # 下面这些是“可在终端修改的参数”；未传入时使用 default 默认值
    parser = argparse.ArgumentParser(description="RoboHeard six-channel WPE test")
    parser.add_argument("--input", default="data/raw/Audio_AEC/S150.wav", help="6-channel AEC WAV")
    parser.add_argument("--start", type=float, default=40.0, help="clip start time (seconds)")
    parser.add_argument("--seconds", type=float, default=10.0, help="clip duration (seconds)")
    parser.add_argument("--ref-channel", type=int, default=1, help="reference mic (1 to 6)")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--taps", type=int, default=5)
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output-root", default="data/processed")
    args = parser.parse_args()  # 收集用户实际传入的参数

    # 基础安全检查：文件是否存在、时间是否合理、参数是否合法
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Audio file not found: {source.resolve()}")
    if args.seconds <= 0 or args.start < 0:
        parser.error("--seconds must be positive and --start must be nonnegative")
    if args.n_fft < 64 or args.hop < 1 or args.hop > args.n_fft:
        parser.error("Invalid STFT settings: require 64 <= n_fft and 1 <= hop <= n_fft")
    if args.taps < 1 or args.delay < 1 or args.iterations < 1:
        parser.error("--taps, --delay and --iterations must be positive")
    if not 1 <= args.ref_channel <= 6:
        parser.error("--ref-channel must be between 1 and 6")

    # 只读取 WAV 文件头，检查采样率和通道数；暂不读取整段十分钟的音频
    info = sf.info(source)
    if info.samplerate != 16000 or info.channels != 6:
        parser.error(f"Expected 16 kHz, 6 channels; got {info.samplerate} Hz, {info.channels} channels")
    # 把“秒”转换成“采样点”：例如 40 秒 × 16000 Hz = 第 640000 个采样点
    start_frame = round(args.start * info.samplerate)
    frame_count = round(args.seconds * info.samplerate)
    if start_frame >= info.frames:
        parser.error(f"Start time {args.start}s is outside the audio (duration {info.duration:.2f}s)")

    # 只截取指定片段，避免一次把整份大音频加载到内存
    with sf.SoundFile(source) as f:
        f.seek(start_frame)
        # audio 形状：[采样点数, 6]；10 秒时约为 [160000, 6]
        audio = f.read(frames=frame_count, dtype="float64", always_2d=True)
    if audio.shape[0] < args.n_fft * 4:
        parser.error("Selected clip is too short; choose at least a second")

    print(f"Input: {source} | {info.samplerate} Hz | {info.channels} channels | full {info.duration:.1f}s")
    print(f"Clip: start={args.start:.2f}s, length={len(audio)/info.samplerate:.2f}s")
    print("Computing multichannel STFT ...", flush=True)
    # 第一步：六通道时域波形 → 六通道 STFT 复数频谱
    spec = multi_stft(audio, info.samplerate, args.n_fft, args.hop)
    print(f"STFT shape [freq, channels, frames]: {spec.shape}", flush=True)
    print(f"Running WPE (taps={args.taps}, delay={args.delay}, iterations={args.iterations}) ...", flush=True)
    # 第二步：在频域执行 WPE，联合六通道估计并抑制“晚期混响”
    # taps：使用多少个历史帧；delay：跳过最近几个帧；iterations：重复更新几轮
    enhanced_spec = wpe_v8(spec, taps=args.taps, delay=args.delay, iterations=args.iterations)
    # 第三步：WPE 后的频谱 → 六通道时域波形，采样点数与输入保持一致
    enhanced = multi_istft(enhanced_spec, info.samplerate, args.n_fft, args.hop, len(audio))
    # 数值安全检查：确保 WPE 没有产生 NaN 或无穷大
    if not np.isfinite(enhanced).all():
        raise RuntimeError("WPE returned NaN or Inf values. Try another clip or adjust WPE parameters.")

    # Apply a common gain only when needed to avoid clipping. Never normalize each mic independently.
    # 防止导出 16-bit PCM 时削波失真；输入和输出使用“同一个”增益
    # 注意：这只是防削波，不能说明去混响的音质一定变好
    peak = float(max(np.max(np.abs(audio)), np.max(np.abs(enhanced))))
    gain = min(1.0, 0.98 / peak) if peak > 0 else 1.0
    if gain < 1.0:
        print(f"Warning: peak {peak:.3f} exceeds safe PCM range; applying common gain {gain:.4f}")
    audio = audio * gain
    enhanced = enhanced * gain

    # 根据命令行输出目录，准备六通道和单通道两类输出文件夹
    out_root = Path(args.output_root)
    out6 = out_root / "wpe_6ch"
    out1 = out_root / "mono_16k"
    out6.mkdir(parents=True, exist_ok=True)
    out1.mkdir(parents=True, exist_ok=True)
    # 文件名包含会话编号、开始时间与片段长度，避免覆盖不同实验
    slug = f"{source.stem}_s{args.start:g}_d{len(audio)/info.samplerate:g}"
    # 命令行通道编号是 1~6；NumPy 的数组下标从 0 开始，因此要减 1
    ref = args.ref_channel - 1
    paths = {
        "six_ch_wpe": out6 / f"{slug}_wpe_6ch.wav",
        "mic_wpe": out1 / f"{slug}_wpe_mic{args.ref_channel}.wav",
        "mic_original": out1 / f"{slug}_aec_mic{args.ref_channel}.wav",
    }
    # 保存三个文件：WPE 后六通道、WPE 后参考通道、AEC 原始参考通道
    # “最佳通道”尚未自动筛选，此处默认 Mic 1 只是供试听的临时参考
    sf.write(paths["six_ch_wpe"], enhanced, info.samplerate, subtype="PCM_16")
    sf.write(paths["mic_wpe"], enhanced[:, ref], info.samplerate, subtype="PCM_16")
    sf.write(paths["mic_original"], audio[:, ref], info.samplerate, subtype="PCM_16")

    # RMS 是信号均方根幅度，可用于观察整体能量变化，但并非音质指标
    rms_before = float(np.sqrt(np.mean(audio[:, ref] ** 2)))
    rms_after = float(np.sqrt(np.mean(enhanced[:, ref] ** 2)))
    print(f"Reference channel: Mic {args.ref_channel} (temporary; not necessarily best)")
    print(f"Ref mic RMS: before={rms_before:.5f}, after={rms_after:.5f} (not a quality score)")
    print("Saved outputs:")
    for label, file in paths.items():
        print(f"  {label:12s} -> {file.resolve()}")
    print("Done. Listen to AEC and WPE reference mic files before choosing full-dataset settings.")


# 只有直接运行此脚本时才执行 main()；被其他模块 import 时不会自动处理音频
if __name__ == "__main__":
    main()
