# RoboHeard

> 机器人听觉（Robot Hearing）项目：面向多说话人音频的说话人日志分离（Speaker Diarization），逐步实现“**谁在什么时候说话**”，并为后续语音识别（ASR）与文本关联打下基础。

**项目状态：实验性基线（Baseline）**。目前已实现一版基于 **WavLM-Large + Conformer + 双曲空间分类头（Hyper-SD）** 的说话人活动检测与训练/评估流程；代码可以进行小规模训练和推理，但**尚未获得可靠的正式验证集指标，也未完成 ASR 与最终提交系统**。

仓库地址：<https://github.com/CLiansheng/RoboHeard>

## 1. 当前方案

```text
16 kHz、6 通道 WAV（训练时随机选 1 个通道）
               │
               ▼
      8 秒窗口 / 4 秒步长
               │
               ▼
         WavLM-Large
    （预训练主干暂时冻结）
               │
               ▼
   25 层隐藏状态可学习加权融合
         + Linear 投影
               │
               ▼
         4 层 Conformer
               │
               ▼
       HyperbolicHead
     （7 个说话人槽位）
               │
       ┌───────┴────────┐
       ▼                ▼
 [T, 128] 类别得分   [T, 7] 活动概率
       │
       ▼
   PIT + CrossEntropy
```

- **输入**：每条 8 秒、16 kHz 单通道语音，形状 `[B, 128000]`；单通道由 6 通道原始 WAV 选取，并非要求预先切分或合成为单通道文件。
- **特征**：WavLM 输出约 399 个时间帧，投影后为 `[B, 399, 256]`；Conformer 保持时间维度。
- **输出**：7 个说话人槽位分别给出每帧讲话概率 `[B, 399, 7]`；内部使用 `2^7 = 128` 种活动组合类别。
- **监督**：通过 JSON 中的 `speaker_id`、`start`、`end` 生成帧级多标签；用排列不变训练（PIT）解决预测槽位与人工标注编号不同的问题。
- **重叠语音**：允许同一帧多个说话人同时活动。

> 7 个槽位来自当前标注中单场 Session 最多出现 7 位不同说话人的统计；**7 个槽位不是 7 个跨 Session 固定的人物身份**。该双曲分类头是工程基线，不能视为 TellWhisper 论文的逐项精确复现。

## 2. 项目结构

```text
RoboHeard/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── diarization/
│       ├── dataset.py           # 8 秒切片、通道增强、JSON → 帧级标签
│       ├── wavlm_encoder.py     # WavLM-Large 与层融合
│       ├── conformer.py         # Conformer 编码器
│       ├── hyperbolic_head.py   # 双曲原型分类器（7 人/128 类）
│       ├── model.py             # 完整模型组装
│       ├── loss.py              # PIT 组合类别交叉熵
│       ├── trainer.py           # 训练、验证、Checkpoint 保存
│       └── evaluate.py          # 窗口级与 Session 级 DER（近似）
├── scripts/
│   ├── compare_channels.py     # 实验性通道对比脚本
│   └── test_wpe.py             # 实验性 WPE 脚本（非正式训练流程）
├── data/                        # 本地自建，不提交 Git
│   └── raw/
│       ├── Audio_AEC/           # S001.wav、S002.wav ...
│       └── DOC_AEC_JSON/        # S001.json、S002.json ...
├── checkpoints/                 # 本地自动生成，不提交 Git
│   ├── pretrained/              # WavLM 预训练权重缓存
│   └── diarization/            # best.pt、last.pt
└── results/                     # 评估输出，不提交 Git
    └── diarization/
        └── eval.json
```

> `data/`、`checkpoints/`、`results/` 是运行时目录，不一定在 GitHub 仓库中显示。

## 3. 队友快速开始（Windows / PowerShell）

### 第一步：克隆仓库

```powershell
git clone https://github.com/CLiansheng/RoboHeard.git
cd RoboHeard
```

### 第二步：创建 Python 环境

建议使用 **Python 3.11**：

```powershell
conda create -n roboheard python=3.11 -y
conda activate roboheard
python -m pip install -r requirements.txt
```

`requirements.txt` 当前为 **PyTorch 2.9.1 + torchaudio 2.9.1（cu128）**、`transformers==4.57.6`、NumPy、SciPy 和 SoundFile。建议使用配备兼容 NVIDIA GPU 的机器；如果设备或驱动不支持该 CUDA 构建，请先根据本机环境选择相匹配的 PyTorch / torchaudio 版本，且两者版本需保持匹配。

检查环境：

```powershell
python -c "import torch, torchaudio, transformers, soundfile, scipy; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('transformers:', transformers.__version__)"
```

`CUDA: True` 表示 PyTorch 可以检测到 CUDA 设备。第一次创建模型时，会从 Hugging Face 下载 `microsoft/wavlm-large` 预训练权重并缓存于 `checkpoints/pretrained/`，需要可用的下载网络和磁盘空间。

### 第三步：放置数据（单独共享，不通过 GitHub）

请从团队共享网盘获取 **Audio_AEC** 音频及与其匹配的 JSON，放置为：

```text
data/raw/Audio_AEC/S150.wav
data/raw/DOC_AEC_JSON/S150.json
```

可按相同规则添加其他 Session，**文件名必须对应**。当前 `dataset.py` 要求每个 WAV 为 **16 kHz、6 通道**；训练时随机取一个通道，验证和评估时使用通道 0。

JSON 示例（仅说明结构）：

```json
{
  "session_id": "S150",
  "duration": 600.0,
  "segments": [
    { "speaker_id": "P054", "start": 0.68, "end": 1.53, "text": "示例文本" }
  ]
}
```

`duration`、说话时间与文本仅作格式示例；以实际标注为准。代码主要使用 `session_id` 与每个片段的 `speaker_id/start/end`。**不要将比赛数据、人物信息或大型模型权重提交到公开仓库；先核对比赛的数据使用与公开规则。**

### 第四步：检查各模块

从仓库根目录依次执行：

```powershell
python -m src.diarization.dataset
python -m src.diarization.wavlm_encoder
python -m src.diarization.conformer
python -m src.diarization.hyperbolic_head
python -m src.diarization.loss
python -m src.diarization.model
```

注意：`dataset.py` 测试默认需要已放置相应 JSON；`wavlm_encoder.py` 和 `model.py` 首次运行可能触发 WavLM 下载。

### 第五步：先跑 3 个 Batch 的训练烟雾测试

```powershell
python -m src.diarization.trainer --epochs 1 --max-train-batches 3 --log-every 1
```

此步骤只检查数据读取、模型前向、PIT Loss、反向传播以及权重保存，不代表模型达到可用精度。单 Session 时没有独立验证集，通常只产生 `last.pt`。

如果流程通过，并且准备了**多个不同 Session** 的数据，可以开始完整轮次的基线训练：

```powershell
python -m src.diarization.trainer --epochs 1 --log-every 50
```

随后根据验证集情况再决定是否运行更多 Epoch，例如：

```powershell
python -m src.diarization.trainer --epochs 10
```

默认训练配置：8 秒窗口、4 秒步长、`batch_size=1`、`grad_accum=8`、AdamW、学习率 `1e-4`、随机种子 42、按 Session 划分训练/验证集（默认验证比例 15%）。有多个 Session 时会保存验证 Loss 最低的 `best.pt` 和当轮的 `last.pt`。

**注意**：再次执行训练命令会重新初始化模型与优化器，目前不支持自动从 `last.pt` 断点续训。

### 第六步：计算 DER

已有训练检查点后，可执行：

```powershell
# 指定一场会话，排查整场推理与跨窗口拼接
python -m src.diarization.evaluate --session S150

# 验证集：需至少两个 Session，并与训练时的 seed/val-ratio 一致
python -m src.diarization.evaluate --split val

# 如需指定权重
python -m src.diarization.evaluate --split val --checkpoint checkpoints/diarization/best.pt
```

默认优先加载 `checkpoints/diarization/best.pt`，不存在时尝试 `last.pt`。详细报告写入 `results/diarization/eval.json`。

**两个 DER 结果不要混淆：**

- **窗口级 DER**：每个 8 秒窗口独立评估，重叠区域可能重复计入，用于诊断。
- **Session 级 DER**：利用相邻窗口重叠部分匹配说话人轨道，整场每帧只计算一次，更接近目标任务，但当前拼接算法仍是启发式方法。

当前计算采用约 20 ms 的特征帧、无 collar、保留重叠讲话；**该分数不是已经核对过官方比赛评分规则的正式 DER**。DER 可大于 100%，并非程序一定算错。

## 4. 当前已知问题与工作进度

**已完成（第一版代码）**：音频切片及标签对齐、WavLM 特征提取、Conformer、7 人双曲分类头、PIT Loss、训练脚本与近似 DER 评估。

**仍需完成或核实**：

1. **正式数据训练**：目前的小数据训练与 DER 只是检查流程，尚未获得稳定的独立验证集指标。
2. **跨窗口身份一致性**：整场拼接依赖重叠时间段；身份交换或语音不重叠时可能出现大量碎片轨道。
3. **预测塌缩与阈值**：需要检查 7 个槽位的激活率、预测人数、阈值 0.5 是否合适，以及静音/重叠比例。
4. **数据覆盖和对齐**：训练 Dataset 对长 WAV 的末尾残余片段尚未完全覆盖；短音频补零与 WavLM Attention Mask 的严格处理仍可完善。
5. **实验指标**：后续增加可靠的 Session 级 DER、错误分析和对照实验（例如欧氏分类头 vs 双曲分类头）。
6. **最终任务**：项目当前主要完成 diarization 模块；ASR、文字与说话人/时间关联、完整竞赛输出尚待开发。

**已观察到的现象**：少量训练 Batch 可以使训练 Loss 下降，但用少量训练得到的权重评估 S150 曾出现很高的 DER 和轨道碎片化。因此**不能把测试跑通或 Loss 下降视为模型已经训练成功**。

## 5. 推荐团队协作方式

- 在 `main` 保持可运行的基线；针对功能改动从 `main` 新建分支，如 `feature/data-validation`、`feature/der-eval`。
- 通过 Pull Request 合并改动，描述改动目的、运行命令、关键实验输出和可能的兼容性问题。
- 不提交原始音频、JSON 数据集、Checkpoint、日志或压缩包；共享数据使用团队认可且符合比赛规则的外部渠道。
- 训练记录至少包含：代码提交 SHA、Session 划分、训练命令、Epoch、训练/验证 Loss、DER、显卡与显存、阈值。

示例日常同步命令：

```powershell
git pull origin main
git checkout -b feature/my-task
# 修改并自测之后
git add src/diarization/
git commit -m "Describe my change"
git push -u origin feature/my-task
```

然后在 GitHub 上发起 Pull Request。不要直接将不同机器上生成的模型检查点互相覆盖；需要共享权重时请标明实验版本。

## 6. 参考与说明

- 预训练声学模型：[`microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large)
- PyTorch：<https://pytorch.org/>
- 参考研究方向：TellWhisper / 双曲空间说话人活动建模。当前项目为相关思路的**工程探索版**，不宣称逐项复现原论文。

> 当前 README 描述的是仓库截至现阶段的基线实现。建议每次修改数据结构、训练参数、模型输出或评估协议时同步更新本文件。
