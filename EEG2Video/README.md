根据你提供的 `/EEG2Video/EEG2Video_New` 文件夹结构，里面包含如下子文件夹和文件：

- `Seq2Seq/`
- `Semantic/`
- `Generation/`
- `DANA/`
- `README.md`

下面我将分别解释这些部分的作用，并给出推荐的阅读顺序。

***

## 各部分作用

### 1. `Seq2Seq/`

- **作用**：通常 `Seq2Seq`（Sequence-to-Sequence）用于序列建模，比如文本生成、机器翻译等。在 EEG2Video 相关项目中，这部分很可能用于将 EEG 信号序列映射为视频帧序列或中间语义序列。
- **内容**：你可以重点关注 `my_autoregressive_transformer.py`，这可能是实现自回归序列模型的核心代码。

### 2. `Semantic/`

- **作用**：`Semantic` 目录一般用于语义相关的处理，比如将 EEG 信号转化为语义标签、文本描述等。这里的 `eeg_text.py` 可能是将 EEG 信号转为文本描述的脚本。
- **内容**：如果你关心 EEG 到语义/文本的映射，这部分值得深入阅读。

### 3. `Generation/`

- **作用**：`Generation` 目录通常包含生成相关的代码，比如视频生成、图像生成等。这里面可能包含了模型定义、推理脚本、训练脚本等。
- **内容**：你可以看到有 `inference_eeg2video.py`（推理脚本）、`models/`（模型定义）、`pipelines/`（生成流程）、`train_finetune_videodiffusion.py`（训练脚本）等，是整个 EEG2Video 生成的核心部分。

### 4. `DANA/`

- **作用**：`DANA` 可能是某个特定模块或方法的实现，比如数据增强（Data Augmentation）、特定的网络结构等。需要结合里面的代码进一步确认。
- **内容**：如 `add_noise.py` 可能是对 EEG 信号或生成数据加噪声的脚本。

### 5. `README.md`

- **作用**：一般会对整个文件夹的结构和用法做简要说明。建议首先阅读。

***

## 推荐阅读顺序

1. **README.md**  
   先看根目录下的 `README.md`，了解整体设计和各部分的简要说明。
2. **Generation/**  
   这是核心生成模块，建议先看 `Generation/README.md`（如果有），再看 `inference_eeg2video.py`（推理流程），`models/`（模型结构），`pipelines/`（生成流程），最后看训练脚本。
3. **Semantic/**  
   如果你对 EEG 到语义/文本的映射感兴趣，可以继续看 `Semantic/` 目录，了解如何从 EEG 信号提取语义信息。
4. **Seq2Seq/**  
   如果你对序列建模、EEG 到视频帧的序列映射感兴趣，可以看 `Seq2Seq/` 目录。
5. **DANA/**  
   最后看 `DANA/`，了解是否有特殊的数据处理或网络结构。

***

## 总结

- **先看 README.md**，把握全局。
- **主线是 Generation/**，这里是 EEG 到视频的生成主流程。
- **Semantic/ 和 Seq2Seq/** 是辅助或中间步骤，分别处理语义和序列建模。
- **DANA/** 是特定方法或工具模块，最后看。

如果你需要更详细的每个子目录内部文件说明，可以告诉我具体想看哪个目录，我可以帮你进一步分析！