![image-20250312094456049](D:\Typora-notes\attachment\image-20250312094456049.png)

---

根据我们之前的对话和对项目结构的理解，以下是您工作区中的一些关键代码文件：

- `requirements.txt` ：此文件列出了项目的所有依赖项，对于理解项目所需的库至关重要。
- `models.py` ：此文件包含了各种 EEG 编码器（如 PatchEmbedding 、 tsconv 、 deepnet 、 eegnet 、 shallownet ）以及我们最近移动过来的 CLIP 类，这些都是模型的核心组件。
- `pipeline_tuneeeg2video.py` ：此文件定义了 TuneAVideoPipeline 类，它是视频生成管道的核心，包含了 EEG 信号编码和潜在表示解码的逻辑。
- `train_semantic_predictor.py` ：尽管 CLIP 类已移出，此文件仍然包含训练 CLIP 模型的完整流程，包括数据加载、模型初始化和训练循环。
这些文件共同构成了项目的主要功能，涵盖了 EEG 信号处理、语义预测和视频生成的核心逻辑。

