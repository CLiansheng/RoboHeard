
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicHead(nn.Module):
    """
    双曲空间说话人活动分类器。

    输入:
        features: [B, T, 256]

    输出:
        class_logits: [B, T, 128]
        speaker_probs: [B, T, 7]

    7 个说话人槽位，对应 2^7 = 128 种活动组合。
    """

    def __init__(
        self,
        input_dim=256,
        embedding_dim=128,
        num_speakers=7,
        curvature=1.0,
        clip_norm=2.0,
    ):
        super().__init__()

        if not 1 <= num_speakers <= 8:
            raise ValueError("说话人槽位数必须在 1～8 之间")

        if curvature <= 0 or clip_norm <= 0:
            raise ValueError("曲率和裁剪半径必须大于 0")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers

        # 7 人对应 128 种说话状态组合
        self.num_classes = 2 ** num_speakers

        self.curvature = curvature
        self.clip_norm = clip_norm
        self.eps = 1e-6

        # 1. 将 Conformer 特征从 256 维映射为 128 维
        self.projection = nn.Linear(
            input_dim,
            embedding_dim,
        )

        # 2. 为每种活动组合定义一个可学习原型
        # 参数存储在欧氏切空间，前向时映射到双曲空间
        self.prototype_tangent = nn.Parameter(
            torch.randn(
                self.num_classes,
                embedding_dim,
            ) * 0.02
        )

        # 3. 构建类别与说话人的对应关系
        # 例如类别 5 (二进制 0000101)
        # 表示第 1 位和第 3 位说话人同时活动
        class_ids = torch.arange(self.num_classes)

        speaker_bits = (
            1 << torch.arange(num_speakers)
        )

        class_to_speaker = (
            (
                class_ids[:, None]
                & speaker_bits[None, :]
            ) != 0
        ).float()

        # [128, 7]，固定映射关系，不参与训练
        self.register_buffer(
            "class_to_speaker",
            class_to_speaker,
        )

    def _clip_tangent(self, x):
        """
        裁剪向量范数，避免映射后过于接近双曲球边界。
        """
        norm = torch.linalg.vector_norm(
            x,
            dim=-1,
            keepdim=True,
        )

        factor = (
            self.clip_norm
            / norm.clamp_min(self.eps)
        ).clamp(max=1.0)

        return x * factor

    def _expmap0(self, x):
        """
        将切空间向量映射到庞加莱球。
        """
        sqrt_c = math.sqrt(self.curvature)

        norm = torch.linalg.vector_norm(
            x,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        mapped = (
            torch.tanh(sqrt_c * norm)
            * x
            / (sqrt_c * norm)
        )

        # 保证点严格位于庞加莱球内部
        limit = (1.0 - 1e-4) / sqrt_c

        mapped_norm = torch.linalg.vector_norm(
            mapped,
            dim=-1,
            keepdim=True,
        )

        factor = (
            limit
            / mapped_norm.clamp_min(self.eps)
        ).clamp(max=1.0)

        return mapped * factor

    def _poincare_distance(self, x, p):
        """
        计算庞加莱球中的双曲距离。

        x: [B, T, 1, D]
        p: [1, 1, 128, D]

        输出: [B, T, 128]
        """
        c = self.curvature

        diff_norm = torch.linalg.vector_norm(
            x - p,
            dim=-1,
        )

        x_norm_sq = (x * x).sum(dim=-1)
        p_norm_sq = (p * p).sum(dim=-1)

        denominator = (
            (1.0 - c * x_norm_sq)
            * (1.0 - c * p_norm_sq)
        ).clamp_min(self.eps)

        distance = (
            2.0 / math.sqrt(c)
            * torch.asinh(
                math.sqrt(c)
                * diff_norm
                / torch.sqrt(denominator)
            )
        )

        return distance

    def forward(self, features):
        """
        features: [B, T, 256]
        """

        # 检查输入形状
        if (
            features.ndim != 3
            or features.shape[-1] != self.input_dim
        ):
            raise ValueError(
                f"输入必须是 [B, T, {self.input_dim}]"
            )

        # 1. 投影到 128 维切空间
        tangent = self.projection(features).float()

        # 2. 限制向量范数
        tangent = self._clip_tangent(tangent)

        # 3. 将帧特征映射到双曲空间
        frame_points = self._expmap0(tangent)

        # 4. 将 128 个可学习原型映射到双曲空间
        prototype_points = self._expmap0(
            self._clip_tangent(
                self.prototype_tangent.float()
            )
        )

        # 5. 计算每帧与 128 个原型的双曲距离
        distances = self._poincare_distance(
            frame_points.unsqueeze(2),
            prototype_points[None, None, :, :],
        )

        # 6. 距离越近，类别得分越高
        class_logits = -distances

        # 7. Softmax 得到 128 种组合的概率
        class_probs = torch.softmax(
            class_logits,
            dim=-1,
        )

        # 8. 将组合概率转换成 7 个说话人的活动概率
        speaker_probs = (
            class_probs @ self.class_to_speaker
        )

        return {
            "class_logits": class_logits,
            "speaker_probs": speaker_probs,
        }


def activity_to_class_index(activities):
    """
    将 7 维说话人活动标签转换成组合类别索引。

    例如:
        [0,0,0,0,0,0,0] -> 0
        [1,0,0,0,0,0,0] -> 1
        [0,1,0,0,0,0,0] -> 2
        [1,1,0,0,0,0,0] -> 3
        [1,0,1,0,0,0,0] -> 5
    """
    bits = 1 << torch.arange(
        activities.shape[-1],
        device=activities.device,
    )

    return (
        (activities > 0.5).long() * bits
    ).sum(dim=-1)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备:", device)

    # 模拟 Conformer 的输出
    features = torch.randn(
        1, 399, 256,
        device=device,
    )

    # 创建 7 人版双曲分类器
    head = HyperbolicHead(
        input_dim=256,
        embedding_dim=128,
        num_speakers=7,
    ).to(device)

    # 前向传播
    outputs = head(features)

    class_logits = outputs["class_logits"]
    speaker_probs = outputs["speaker_probs"]

    print("输入形状:", tuple(features.shape))
    print("类别得分:", tuple(class_logits.shape))
    print("说话人概率:", tuple(speaker_probs.shape))

    # 验证输出形状
    assert class_logits.shape == (1, 399, 128)
    assert speaker_probs.shape == (1, 399, 7)

    # 检查输出是否为正常数值和概率
    assert torch.isfinite(class_logits).all()
    assert torch.isfinite(speaker_probs).all()

    assert (
        (speaker_probs >= 0)
        & (speaker_probs <= 1)
    ).all()

    # 构造模拟的七人活动标签
    labels = torch.zeros(
        1, 399, 7,
        device=device,
    )

    # 第 1 位说话人持续讲话
    labels[:, :, 0] = 1

    # 第 3 位说话人在第 100～149 帧同时讲话
    labels[:, 100:150, 2] = 1

    # 将多标签状态编码成类别编号
    class_targets = activity_to_class_index(
        labels
    )

    # 模拟训练 Loss
    loss = F.cross_entropy(
        class_logits.reshape(-1, 128),
        class_targets.reshape(-1),
    )

    # 测试反向传播
    loss.backward()

    assert head.projection.weight.grad is not None
    assert head.prototype_tangent.grad is not None

    assert torch.isfinite(
        head.projection.weight.grad
    ).all()

    assert torch.isfinite(
        head.prototype_tangent.grad
    ).all()

    print("模拟 Loss:", loss.item())
    print("7 人版 HyperbolicHead 测试通过！")
