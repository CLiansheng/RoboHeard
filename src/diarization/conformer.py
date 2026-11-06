
import torch
import torch.nn as nn
from torchaudio.models import Conformer


class ConformerEncoder(nn.Module):
    """
    RoboHeard 的 Conformer 编码器。

    输入:  [Batch, Frames, 256]
    输出:  [Batch, Frames, 256]

    作用:
    进一步提取局部声学信息和长距离上下文信息。
    """

    def __init__(
        self,
        input_dim=256,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim

        # 创建四层串行的 Conformer
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

    def forward(self, features, lengths=None):
        """
        features: [B, T, 256]
        lengths:  [B]，每条音频的有效特征帧数
        """

        if features.ndim != 3:
            raise ValueError("输入必须是 [Batch, Frames, Features]")

        if features.shape[-1] != self.input_dim:
            raise ValueError("输入特征维度不正确")

        batch_size, num_frames, _ = features.shape

        # 如果没有提供有效长度，就认为全部帧有效
        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                num_frames,
                dtype=torch.long,
                device=features.device,
            )
        else:
            lengths = lengths.to(
                device=features.device,
                dtype=torch.long,
            )

            if lengths.shape != (batch_size,):
                raise ValueError("lengths 的形状必须是 [Batch]")

        # Conformer 前向传播
        output, output_lengths = self.conformer(
            features,
            lengths,
        )

        return output, output_lengths


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备:", device)

    # 模拟 WavLM-Large 的输出特征
    features = torch.randn(1, 399, 256).to(device)

    print("输入形状:", tuple(features.shape))

    # 创建 Conformer
    encoder = ConformerEncoder().to(device)
    encoder.eval()

    # 测试前向传播
    with torch.no_grad():
        output, lengths = encoder(features)

    print("输出形状:", tuple(output.shape))
    print("有效帧数:", lengths.tolist())

    assert output.shape == features.shape
    assert torch.isfinite(output).all()

    print("Conformer 测试通过！")
