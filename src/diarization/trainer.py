
"""RoboHeard Hyper-SD 第一版训练脚本。"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import RoboHeardDataset
from .loss import PITCombinationLoss
from .model import HyperSDModel


def set_seed(seed):
    """设置随机种子，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args, device):
    """按 Session 划分训练集和验证集。"""

    audio_dir = Path(args.audio_dir)
    annotation_dir = Path(args.annotation_dir)

    if not audio_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            "音频目录或 JSON 标注目录不存在"
        )

    session_ids = sorted(
        p.stem for p in audio_dir.glob("*.wav")
    )

    if not session_ids:
        raise RuntimeError(
            f"未在 {audio_dir} 找到 WAV 文件"
        )

    missing = [
        sid for sid in session_ids
        if not (
            annotation_dir / f"{sid}.json"
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"这些会话缺少 JSON: {missing[:10]}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(session_ids)

    # 按整个 Session 划分，避免数据泄漏
    if len(session_ids) == 1 or args.val_ratio == 0:
        train_ids = session_ids
        val_ids = []

        print(
            "仅进行训练测试，不计算验证 Loss。"
        )

    else:
        val_count = max(
            1,
            round(
                len(session_ids) * args.val_ratio
            ),
        )

        val_count = min(
            val_count,
            len(session_ids) - 1,
        )

        val_ids = session_ids[:val_count]
        train_ids = session_ids[val_count:]

    print(f"训练 Session: {len(train_ids)} 个")
    print(f"验证 Session: {len(val_ids)} 个")

    print("训练 Session 示例:", train_ids[:5])
    print("验证 Session 示例:", val_ids[:5])

    common = dict(
        audio_dir=str(audio_dir),
        annotation_dir=str(annotation_dir),
        chunk_duration=8.0,
        stride=4.0,
        sample_rate=16000,
        num_speakers=7,
    )

    train_dataset = RoboHeardDataset(
        **common,
        mode="train",
        session_ids=train_ids,
    )

    if not len(train_dataset):
        raise RuntimeError("训练数据集为空")

    val_dataset = (
        RoboHeardDataset(
            **common,
            mode="val",
            session_ids=val_ids,
        )
        if val_ids
        else None
    )

    loader_args = dict(
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            shuffle=False,
            **loader_args,
        )
        if val_dataset is not None
        else None
    )

    return train_loader, val_loader


def move_batch(batch, device):
    """将一个 Batch 移动到 CPU/GPU。"""

    non_blocking = device.type == "cuda"

    audio = batch["audio"].to(
        device,
        non_blocking=non_blocking,
    )

    labels = batch["labels"].to(
        device,
        non_blocking=non_blocking,
    )

    valid_frames = batch["valid_frames"].to(
        device,
        non_blocking=non_blocking,
    )

    return audio, labels, valid_frames


def compute_loss(
    model,
    criterion,
    batch,
    device,
    use_amp,
    amp_dtype,
):
    """模型前向传播，并计算 PIT Loss。"""

    audio, labels, valid_frames = move_batch(
        batch,
        device,
    )

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            audio,
            valid_frames=valid_frames,
        )

        if (
            outputs["class_logits"].shape[:2]
            != labels.shape[:2]
        ):
            raise RuntimeError(
                "模型输出的时间帧数与 JSON 标签不一致："
                f"{outputs['class_logits'].shape}"
                f" vs {labels.shape}"
            )

        loss = criterion(
            outputs["class_logits"],
            labels,
            valid_frames=valid_frames,
        )

    return loss


def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    trainable_params,
    args,
    device,
    use_amp,
    amp_dtype,
):
    """训练一个 Epoch。"""

    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_batches = len(loader)

    if args.max_train_batches > 0:
        total_batches = min(
            total_batches,
            args.max_train_batches,
        )

    total_loss = 0.0

    for batch_index, batch in enumerate(loader):

        if batch_index >= total_batches:
            break

        loss = compute_loss(
            model,
            criterion,
            batch,
            device,
            use_amp,
            amp_dtype,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"训练 Loss 非有限值: {loss.item()}"
            )

        # 计算本组实际的梯度累积次数
        group_start = (
            batch_index // args.grad_accum
        ) * args.grad_accum

        group_size = min(
            args.grad_accum,
            total_batches - group_start,
        )

        # 梯度累积
        scaler.scale(
            loss / group_size
        ).backward()

        total_loss += loss.detach().item()

        # 每累计指定次数，更新一次模型
        should_step = (
            (batch_index + 1) % args.grad_accum == 0
            or batch_index + 1 == total_batches
        )

        if should_step:

            scaler.unscale_(optimizer)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                args.grad_clip,
            )

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        if (
            (batch_index + 1) % args.log_every == 0
            or batch_index + 1 == total_batches
        ):
            print(
                f"  train batch "
                f"{batch_index + 1}/{total_batches} "
                f"loss={loss.item():.4f}"
            )

    return total_loss / total_batches


@torch.no_grad()
def validate(
    model,
    criterion,
    loader,
    args,
    device,
    use_amp,
    amp_dtype,
):
    """验证模型，不更新参数。"""

    model.eval()

    total_batches = len(loader)

    if args.max_val_batches > 0:
        total_batches = min(
            total_batches,
            args.max_val_batches,
        )

    total_loss = 0.0

    for batch_index, batch in enumerate(loader):

        if batch_index >= total_batches:
            break

        loss = compute_loss(
            model,
            criterion,
            batch,
            device,
            use_amp,
            amp_dtype,
        )

        total_loss += loss.item()

    return total_loss / total_batches



def save_trainable_checkpoint(
    model,
    path,
    epoch,
    metric,
    args,
):
    """
    保存 Hyper-SD 模型状态。

    保存:
        WavLM 层融合权重和输出投影
        Conformer 所有参数及 BatchNorm 状态
        双曲分类器所有参数和原型

    不保存:
        冻结的 WavLM-Large 预训练主干
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. 获取模型全部参数和缓冲状态
    full_state = model.state_dict()

    # 2. 排除冻结的 WavLM-Large 主干
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in full_state.items()
        if not name.startswith("wavlm.wavlm.")
    }

    # 3. 组织检查点内容
    checkpoint = {
        "epoch": epoch,
        "metric": metric,
        "model_state": model_state,
        "model_config": {
            "num_speakers": 7,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "conformer_layers": 4,
        },
        "training_args": vars(args),
    }

    # 4. 保存到磁盘
    torch.save(checkpoint, path)

    print(f"已保存模型: {path}")


def parse_args():
    """读取训练配置。"""

    parser = argparse.ArgumentParser(
        description="RoboHeard Hyper-SD 训练"
    )

    parser.add_argument(
        "--audio-dir",
        default="data/raw/Audio_AEC",
    )

    parser.add_argument(
        "--annotation-dir",
        default="data/raw/DOC_AEC_JSON",
    )

    parser.add_argument(
        "--save-dir",
        default="checkpoints/diarization",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.epochs <= 0
        or args.grad_accum <= 0
        or args.log_every <= 0
    ):
        parser.error(
            "epochs、grad-accum 和 log-every 必须大于 0"
        )

    if not 0 <= args.val_ratio < 1:
        parser.error(
            "val-ratio 必须在 [0, 1) 范围内"
        )

    if args.grad_clip <= 0:
        parser.error(
            "grad-clip 必须大于 0"
        )

    if args.num_workers < 0:
        parser.error(
            "num-workers 不能小于 0"
        )

    return args


def main():

    # 1. 读取配置
    args = parse_args()

    set_seed(args.seed)

    # 2. 确定运行设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # 3. 自动混合精度
    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    amp_dtype = (
        torch.bfloat16
        if (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            use_amp
            and amp_dtype == torch.float16
        ),
    )

    print("运行设备:", device)

    print(
        "混合精度:",
        str(amp_dtype) if use_amp else "关闭",
    )

    # 4. 加载数据
    train_loader, val_loader = build_loaders(
        args,
        device,
    )

    # 5. 创建完整模型
    model = HyperSDModel(
        num_speakers=7,
        hidden_dim=256,
        embedding_dim=128,
        conformer_layers=4,
    ).to(device)

    # 6. 创建 PIT Loss
    criterion = PITCombinationLoss(
        num_speakers=7
    ).to(device)

    # 7. 只优化未冻结的参数
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        "可训练参数量:",
        f"{sum(p.numel() for p in trainable_params):,}",
    )

    best_val = float("inf")
    save_dir = Path(args.save_dir)

    # 8. 开始 Epoch 循环
    for epoch in range(1, args.epochs + 1):

        print(
            f"\n========== "
            f"Epoch {epoch}/{args.epochs} "
            f"=========="
        )

        train_loss = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            trainable_params,
            args,
            device,
            use_amp,
            amp_dtype,
        )

        print(f"train_loss={train_loss:.4f}")

        # 9. 验证
        if val_loader is not None:

            val_loss = validate(
                model,
                criterion,
                val_loader,
                args,
                device,
                use_amp,
                amp_dtype,
            )

            print(f"val_loss={val_loss:.4f}")

            # 保存验证 Loss 最低的模型
            if val_loss < best_val:

                best_val = val_loss

                save_trainable_checkpoint(
                    model,
                    save_dir / "best.pt",
                    epoch,
                    val_loss,
                    args,
                )

        else:
            print(
                "没有独立验证 Session，跳过验证。"
            )

        # 10. 保存当前 Epoch 的参数
        save_trainable_checkpoint(
            model,
            save_dir / "last.pt",
            epoch,
            train_loss,
            args,
        )

        if device.type == "cuda":

            print(
                "GPU 峰值分配显存: "
                f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB"
            )

            torch.cuda.reset_peak_memory_stats()

    print(
        "\n训练完成。"
        "注意：小样本训练测试不能代表真实 DER。"
    )


if __name__ == "__main__":
    main()
