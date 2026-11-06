
from pathlib import Path

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMEncoder(nn.Module):
    """
    RoboHeard 的 WavLM-Large 特征提取器。

    输入:  [Batch, Samples]
    输出:  [Batch, Frames, 256]

    第一阶段:
    - 冻结 WavLM-Large 主干
    - 训练各层融合权重
    - 训练线性投影层
    """

    def __init__(
        self,
        model_name="microsoft/wavlm-large",
        output_dim=256,
    ):
        super().__init__()

        # 1. 加载预训练 WavLM-Large
        # 模型缓存放在项目所在的 D 盘
        cache_dir = Path("checkpoints/pretrained").resolve()

        self.wavlm = WavLMModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
        )

        # 2. 第一阶段冻结 WavLM 的全部参数
        for param in self.wavlm.parameters():
            param.requires_grad_(False)

        self.wavlm.eval()

        # 3. 为每一层设置一个可学习的融合权重
        # WavLM-Large: 1 个初始隐藏状态 + 24 层输出
        num_layers = self.wavlm.config.num_hidden_layers + 1

        self.layer_weights = nn.Parameter(
            torch.zeros(num_layers)
        )

        # 4. 将融合后的 1024 维特征映射到 256 维
        hidden_size = self.wavlm.config.hidden_size

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )

    def train(self, mode=True):
        """
        即使整体模型进入训练模式，
        冻结的 WavLM 仍然保持 eval 模式。
        """
        super().train(mode)
        self.wavlm.eval()
        return self

    def forward(self, waveform):
        """
        waveform: [B, 128000]，8 秒 16kHz 音频
        """

        if waveform.ndim != 2:
            raise ValueError("输入必须是 [Batch, Samples]")

        # 1. 波形标准化
        # 与预训练模型的零均值、单位方差处理保持一致
        waveform = waveform.float()

        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        waveform = (waveform - mean) / torch.sqrt(
            var + 1e-7
        )

        # 2. WavLM 前向传播
        # 冻结主干，不计算其参数梯度
        with torch.no_grad():
            outputs = self.wavlm(
                input_values=waveform,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        # 3. Softmax 归一化各层权重
        weights = torch.softmax(
            self.layer_weights,
            dim=0,
        )

        # 4. 对不同层的特征进行加权求和
        fused = torch.zeros_like(hidden_states[0])

        for weight, hidden in zip(weights, hidden_states):
            fused = fused + weight * hidden

        # 5. 特征维度 1024 -> 256
        features = self.output_projection(fused)

        return features


if __name__ == "__main__":

    from src.diarization.dataset import RoboHeardDataset

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备:", device)

    # 使用之前已经实现的 Dataset
    dataset = RoboHeardDataset(
        audio_dir="data/raw/Audio_AEC",
        mode="val",
        chunk_duration=8.0,
        stride=4.0,
    )

    # 取出一条 8 秒音频
    sample = dataset[0]
    waveform = sample["audio"].unsqueeze(0).to(device)

    print("输入形状:", tuple(waveform.shape))

    # 加载 WavLM-Large
    encoder = WavLMEncoder().to(device)
    encoder.eval()

    # 只测试前向传播
    with torch.no_grad():
        features = encoder(waveform)

    print("输出形状:", tuple(features.shape))
    print("输出数据类型:", features.dtype)

    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[-1] == 256
    assert torch.isfinite(features).all()

    print("WavLM-Large 测试通过！")
