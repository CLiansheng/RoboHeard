
import torch
import torch.nn as nn

from .wavlm_encoder import WavLMEncoder
from .conformer import ConformerEncoder
from .hyperbolic_head import HyperbolicHead


class HyperSDModel(nn.Module):
    """
    RoboHeard 第一版 Hyper-SD 模型。

    网络结构:
        WavLM-Large
            ↓
        层融合 + 线性投影
            ↓
        4 层 Conformer
            ↓
        7 人双曲空间分类器

    输入:
        waveform: [B, Samples]

    输出:
        class_logits: [B, T, 128]
        speaker_probs: [B, T, 7]
        lengths: [B]
    """

    def __init__(
        self,
        num_speakers=7,
        hidden_dim=256,
        embedding_dim=128,
        conformer_layers=4,
    ):
        super().__init__()

        # 1. WavLM-Large 特征提取器
        self.wavlm = WavLMEncoder(
            model_name="microsoft/wavlm-large",
            output_dim=hidden_dim,
        )

        # 2. Conformer 时序编码器
        self.conformer = ConformerEncoder(
            input_dim=hidden_dim,
            num_heads=4,
            ffn_dim=1024,
            num_layers=conformer_layers,
            conv_kernel_size=31,
            dropout=0.1,
        )

        # 3. 双曲空间分类器
        self.classifier = HyperbolicHead(
            input_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_speakers=num_speakers,
        )

    def forward(self, waveform, valid_frames=None):
        """
        waveform: [B, Samples]
        valid_frames: [B] - 每个样本的有效帧数
        """

        if waveform.ndim != 2:
            raise ValueError(
                "输入必须是 [Batch, Samples]"
            )

        # 第一步：WavLM 提取语音特征
        # [B, 128000] -> [B, T, 256]
        features = self.wavlm(waveform)

        # 第二步：Conformer 建模上下文
        # [B, T, 256] -> [B, T, 256]
        features, lengths = self.conformer(features, valid_frames)

        # 第三步：双曲分类器
        # [B, T, 256] -> [B, T, 128] 和 [B, T, 7]
        outputs = self.classifier(features)

        return {
            "class_logits": outputs["class_logits"],
            "speaker_probs": outputs["speaker_probs"],
            "lengths": lengths,
        }


if __name__ == "__main__":

    from .dataset import RoboHeardDataset

    # 1. 选择 GPU 或 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备:", device)

    # 2. 创建测试 Dataset
    dataset = RoboHeardDataset(
        audio_dir="data/raw/Audio_AEC",
        mode="val",
        chunk_duration=8.0,
        stride=4.0,
    )

    # 3. 读取一条 8 秒音频
    sample = dataset[0]

    waveform = (
        sample["audio"]
        .unsqueeze(0)
        .to(device)
    )

    print("输入形状:", tuple(waveform.shape))

    # 4. 创建完整 Hyper-SD
    model = HyperSDModel(
        num_speakers=7,
        hidden_dim=256,
        embedding_dim=128,
        conformer_layers=4,
    ).to(device)

    # 5. 切换到测试模式
    model.eval()

    # 6. 完整模型前向传播
    with torch.inference_mode():
        outputs = model(waveform)

    class_logits = outputs["class_logits"]
    speaker_probs = outputs["speaker_probs"]
    lengths = outputs["lengths"]

    print(
        "类别得分形状:",
        tuple(class_logits.shape)
    )

    print(
        "说话人概率形状:",
        tuple(speaker_probs.shape)
    )

    print("有效帧数:", lengths.tolist())

    # 7. 检查输出
    assert class_logits.ndim == 3
    assert class_logits.shape[-1] == 128

    assert speaker_probs.ndim == 3
    assert speaker_probs.shape[-1] == 7

    assert class_logits.shape[:2] == speaker_probs.shape[:2]

    assert torch.isfinite(class_logits).all()
    assert torch.isfinite(speaker_probs).all()

    assert (
        (speaker_probs >= 0)
        & (speaker_probs <= 1)
    ).all()

    print("完整 Hyper-SD 模型测试通过！")
