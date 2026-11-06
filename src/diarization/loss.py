
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic_head import activity_to_class_index


class PITCombinationLoss(nn.Module):
    """
    7 人活动组合的排列不变交叉熵损失。

    输入:
        class_logits: [B, T, 128]
        labels:       [B, T, 7]
        valid_frames: [B]

    输出:
        loss: 标量
    """

    def __init__(self, num_speakers=7):
        super().__init__()

        self.num_speakers = num_speakers
        self.num_classes = 2 ** num_speakers

        # 1. 枚举所有说话人槽位排列
        permutations = torch.tensor(
            list(
                itertools.permutations(
                    range(num_speakers)
                )
            ),
            dtype=torch.long,
        )

        self.register_buffer(
            "permutations",
            permutations,
        )

        # 2. 生成 128 种活动组合的二进制表示
        class_ids = torch.arange(
            self.num_classes
        )

        bits = (
            class_ids[:, None]
            >> torch.arange(num_speakers)
        ) & 1

        bit_weights = (
            1 << torch.arange(num_speakers)
        )

        # 3. 预计算每个排列下的类别编号变化
        # 例如交换槽位 0、1：
        # 原类别 1 -> 新类别 2
        class_lookup = (
            bits[:, permutations] * bit_weights
        ).sum(dim=-1).T.contiguous()

        # [5040, 128]
        self.register_buffer(
            "class_lookup",
            class_lookup,
        )

    def forward(
        self,
        class_logits,
        labels,
        valid_frames=None,
    ):
        """
        class_logits: [B, T, 128]
        labels:       [B, T, 7]
        """

        if (
            class_logits.ndim != 3
            or class_logits.shape[-1] != self.num_classes
        ):
            raise ValueError(
                "class_logits 的形状不正确"
            )

        batch_size, num_frames, _ = (
            class_logits.shape
        )

        if labels.shape != (
            batch_size,
            num_frames,
            self.num_speakers,
        ):
            raise ValueError(
                "labels 的形状不正确"
            )

        labels = labels.to(
            device=class_logits.device
        )

        # 4. 获取每条样本的有效帧数
        if valid_frames is None:

            valid_frames = torch.full(
                (batch_size,),
                num_frames,
                device=class_logits.device,
                dtype=torch.long,
            )

        else:

            valid_frames = torch.as_tensor(
                valid_frames,
                device=class_logits.device,
                dtype=torch.long,
            ).reshape(-1)

            if (
                valid_frames.numel() != batch_size
                or (valid_frames <= 0).any()
                or (valid_frames > num_frames).any()
            ):
                raise ValueError(
                    "valid_frames 不合法"
                )

        # 5. 生成有效帧 Mask
        frame_indices = torch.arange(
            num_frames,
            device=class_logits.device,
        )

        mask = (
            frame_indices[None, :]
            < valid_frames[:, None]
        )

        # 6. 将七人活动标签转换为组合类别编号
        # [B, T, 7] -> [B, T]
        original_targets = (
            activity_to_class_index(labels)
        )

        # ------------------------------------
        # PIT：寻找 Loss 最小的排列
        # 此阶段无需计算梯度
        # ------------------------------------

        with torch.no_grad():

            # 7. 取得所有排列对应的真实类别
            # [B, 5040, T]
            all_targets = (
                self.class_lookup[
                    :, original_targets
                ].permute(1, 0, 2)
            )

            # 8. 计算各类别的负对数概率
            # [B, 128, T]
            negative_log_probs = (
                -F.log_softmax(
                    class_logits.detach().float(),
                    dim=-1,
                ).transpose(1, 2)
            )

            # 9. 一次性查询全部排列的逐帧 Loss
            # [B, 5040, T]
            all_losses = torch.gather(
                negative_log_probs,
                dim=1,
                index=all_targets,
            )

            # 10. 忽略 Padding，计算平均 Loss
            permutation_losses = (
                (
                    all_losses
                    * mask[:, None, :]
                ).sum(dim=-1)
                / valid_frames[:, None]
            )

            # 11. 每条样本选出 Loss 最小的排列
            # [B]
            best_indices = (
                permutation_losses.argmin(dim=1)
            )

            # 12. 根据最佳排列重新生成真实类别
            # [B, T]
            best_targets = (
                self.class_lookup[best_indices]
                .gather(
                    dim=1,
                    index=original_targets,
                )
            )

        # ------------------------------------
        # 使用最佳排列计算可反向传播的 Loss
        # ------------------------------------

        # 13. 计算逐帧交叉熵
        # [B, T]
        per_frame_loss = F.cross_entropy(
            class_logits.float().transpose(1, 2),
            best_targets,
            reduction="none",
        )

        # 14. 忽略无效帧，取平均
        loss = (
            (per_frame_loss * mask).sum()
            / mask.sum()
        )

        return loss


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备:", device)

    torch.manual_seed(0)

    # 创建 PIT Loss
    criterion = PITCombinationLoss(
        num_speakers=7
    ).to(device)

    print(
        "排列数量:",
        criterion.permutations.shape[0],
    )

    # 模拟一段 20 帧的真实标签
    labels = torch.randint(
        0, 2,
        (1, 20, 7),
        device=device,
    ).float()

    # 故意交换几个说话人槽位
    permutation = torch.tensor(
        [2, 0, 1, 3, 4, 5, 6],
        device=device,
    )

    # 生成与交换后槽位匹配的预测类别
    predicted_targets = activity_to_class_index(
        labels[..., permutation]
    )

    # 构造高置信度预测 Logits
    class_logits = torch.full(
        (1, 20, 128),
        -5.0,
        device=device,
    )

    class_logits.scatter_(
        dim=-1,
        index=predicted_targets.unsqueeze(-1),
        value=5.0,
    )

    class_logits.requires_grad_()

    # 模拟最后 2 帧为 Padding
    valid_frames = torch.tensor(
        [18],
        device=device,
    )

    # 测试 PIT Loss
    loss = criterion(
        class_logits,
        labels,
        valid_frames,
    )

    print("PIT Loss:", loss.item())

    # 测试梯度
    loss.backward()

    assert class_logits.grad is not None
    assert torch.isfinite(
        class_logits.grad
    ).all()

    assert loss.item() < 0.1

    print("PIT Loss 测试通过！")
