好的，我们来详细解释短时傅里叶变换（Short-Time Fourier Transform, STFT），包括其基本原理、核心思想、重要参数，并附上完整的 Python 实现代码和图解。

-----

### 什么是短时傅里叶变换 (STFT)？

标准的傅里叶变换 (FT) 非常适合分析**平稳信号**，即频率成分不随时间变化的信号。它能告诉我们信号中包含哪些频率成分，但无法告诉我们这些频率成分是在哪个时间点出现的。

然而，现实世界中的许多信号（如语音、音乐、生物信号等）都是**非平稳信号**，其频率内容会随着时间演变。例如，一段音乐旋律中，音高（频率）是不断变化的。如果我们对整段音乐做一次傅里叶变换，我们会得到所有音符的频率，但会丢失它们出现的时间顺序信息。

**短时傅里叶变换 (STFT) 就是为了解决这个问题而设计的。**

核心思想很简单：与其一次性对整个信号进行傅里ye变换，不如将信号切分成很多短暂的、可以近似看作平稳的小段，然后分别对每一小段进行傅里叶变换。这样，我们就能得到一系列的频谱图，每一张都对应一个特定的时间点，从而构建出一幅**时间-频率**的联合表示图，即**语谱图 (Spectrogram)**。

### STFT 的工作原理

STFT 的实现过程主要包含以下三个步骤：

1.  **分帧 (Framing) 与加窗 (Windowing)**
2.  **对每一帧进行傅里叶变换 (FFT)**
3.  **拼接结果，形成语谱图**

让我们详细分解每一步：

#### 1\. 分帧与加窗

首先，我们将长信号切割成多个等长的短时帧 (Frame)。这些帧通常会有一部分相互**重叠 (Overlap)**。

  * **帧长 (Frame Length / Window Size)**：这是 STFT 中最关键的参数。它定义了我们一次分析多长的数据。
  * **重叠 (Overlap)**：相邻的两个帧之间会有一部分数据是重叠的。这样做是为了减少由于“帧边界效应”导致的信息丢失，使得时间轴上的过渡更平滑。重叠的长度通常是帧长的一部分，例如 50% 或 75%。与重叠相对应的另一个参数是**步长 (Hop Length)**，即相邻帧的起始点之间的距离。 `Hop Length = Frame Length - Overlap Length`。

在切割出每一帧后，我们会对它乘以一个**窗函数 (Window Function)**，如汉宁窗 (Hann)、汉明窗 (Hamming) 或布莱克曼窗 (Blackman)。

**为什么要加窗？**
直接从信号中“截断”一帧会产生突变的边界，这在频域上会引入不必要的噪声，这种现象称为**频谱泄漏 (Spectral Leakage)**。窗函数的形状是中间高、两边低，它可以平滑帧的两端，使其过渡到零，从而显著减少频谱泄漏，让分析结果更准确地反映该时间段内的真实频率成分。

*图1：STFT的加窗与重叠过程。窗函数（如汉宁窗）在信号上滑动，每次移动一个步长(Hop Length)。*

#### 2\. 对每一帧进行快速傅里叶变换 (FFT)

对加窗后的每一帧数据，我们都执行一次快速傅里叶变换 (FFT)。FFT 的结果是一组复数，代表了在该时间帧内信号的幅度和相位信息。

$$X[k, m] = \sum_{n=0}^{N-1} x[n] w[n-mH] e^{-j \frac{2\pi nk}{N}}$$

其中：

  * $x[n]$ 是原始信号。
  * $w[n]$ 是窗函数。
  * $N$ 是 FFT 的点数（通常等于窗长）。
  * $H$ 是步长 (Hop Length)。
  * $m$ 是帧的索引（代表时间）。
  * $k$ 是频率的索引。

#### 3\. 拼接结果：语谱图 (Spectrogram)

我们将每一帧的 FFT 结果（通常取其幅度的平方或对数值）在时间轴上拼接起来，就形成了一个二维的矩阵。这个矩阵就是 STFT 的结果，可以被可视化为一张图像，即**语谱图 (Spectrogram)**。

  * **X 轴**: 时间
  * **Y 轴**: 频率
  * **颜色/亮度**: 幅度或功率（表示该频率在该时间的能量强弱）

#### 关键参数与权衡：时间-频率分辨率

STFT 的一个核心特性是它受**海森堡-加博尔不确定性原理 (Heisenberg-Gabor Uncertainty Principle)** 的限制。这意味着我们无法同时获得无限高的时间分辨率和频率分辨率。这其中的权衡由**窗长 (Window Size)** 决定：

  * **长窗 (宽窗)**:

      * **优点**: 频率分辨率高。因为分析的信号段更长，可以更精确地区分相近的频率。
      * **缺点**: 时间分辨率低。我们只能模糊地知道这些频率成分存在于这段较长的时间内，无法精确定位其出现瞬间。

  * **短窗 (窄窗)**:

      * **优点**: 时间分辨率高。可以非常精确地定位频率事件发生的时间点。
      * **缺点**: 频率分辨率低。因为分析的信号段太短，根据傅里叶理论，无法分辨出非常接近的频率。

选择合适的窗长是使用 STFT 的关键，需要根据具体的应用场景来决定。

-----

### Python 代码实现

我们将使用 Python 中强大的科学计算库 `NumPy`, `SciPy` 和 `Matplotlib` 来实现和可视化 STFT。`scipy.signal.stft` 是一个非常方便的内置函数。

#### 示例：分析一个频率随时间变化的信号 (Chirp Signal)

我们将创建一个**线性调频信号 (Chirp Signal)**，其频率从低到高线性增加。这种信号是检验 STFT 效果的完美范例。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# 1. 生成一个样本信号
# --------------------
# 我们将创建一个频率随时间线性增加的信号（线性调频信号）
# 这能很好地展示STFT如何捕捉时变的频率

fs = 1000  # 采样频率 (Hz)
T = 5      # 信号时长 (s)
t = np.linspace(0, T, T * fs, endpoint=False) # 时间向量

# 信号由两部分组成:
# a) 前半段 (0-2.5s): 频率从 50Hz 线性增加到 200Hz 的 chirp 信号
# b) 后半段 (2.5-5s): 一个 300Hz 的固定频率正弦波，加上一个 400Hz 的固定频率正弦波
f_start = 50
f_end = 200
t_chirp = np.linspace(0, T/2, int(T/2 * fs), endpoint=False)
chirp_signal = np.sin(2 * np.pi * (f_start * t_chirp + (f_end - f_start) / (T) * t_chirp**2))

t_sin = np.linspace(0, T/2, int(T/2 * fs), endpoint=False)
sin_signal = np.sin(2 * np.pi * 300 * t_sin) + 0.5 * np.sin(2 * np.pi * 400 * t_sin)

# 合并信号
signal = np.concatenate((chirp_signal, sin_signal))

# 2. 执行短时傅里叶变换 (STFT)
# -----------------------------
# 使用 scipy.signal.stft 函数
window_length_samples = 256  # 窗长 (nperseg)
overlap_samples = 128        # 重叠长度 (noverlap)
# Hop length = window_length_samples - overlap_samples = 128

f, t_stft, Zxx = stft(
    signal,
    fs=fs,
    window='hann',              # 使用汉宁窗
    nperseg=window_length_samples,      # 每个段的长度（窗长）
    noverlap=overlap_samples,   # 相邻段之间的重叠样本数
    nfft=None                   # FFT长度，None表示与nperseg相同
)

# Zxx 是一个复数矩阵，包含了幅度和相位信息
# 为了可视化，我们通常取其幅度的对数

# 3. 可视化结果
# ---------------
plt.figure(figsize=(12, 8))

# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (Chirp + Two Sines)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

# 绘制语谱图 (Spectrogram)
plt.subplot(2, 1, 2)
# 使用 pcolormesh 进行绘图，并对幅度取对数以增强可视化效果
# 1e-10 是为了防止 log(0)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('Spectrogram (STFT)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 500) # 限制Y轴范围以便看得更清楚

plt.tight_layout()
plt.show()

```

#### 代码解释

1.  **信号生成**: 我们创建了一个5秒长的信号。前2.5秒，频率从50Hz线性爬升到200Hz。后2.5秒，信号是300Hz和400Hz两个正弦波的叠加。

2.  **`scipy.signal.stft` 函数**:

      * `signal`, `fs`: 输入信号和采样率。
      * `window='hann'`: 指定使用平滑的汉宁窗来减少频谱泄漏。
      * `nperseg=256`: 这是**窗长**，设为256个采样点。这个值决定了频率分辨率。
      * `noverlap=128`: 重叠长度。这里设置了50%的重叠 (`128 / 256 = 0.5`)，意味着步长也是128。
      * 该函数返回三个值：
          * `f`: 频率轴的数组。
          * `t_stft`: 时间轴的数组。
          * `Zxx`: 核心输出，一个二维复数数组，代表每个时间和频率点的STFT结果。

3.  **可视化**:

      * 我们绘制了原始信号以供参考。
      * 使用 `plt.pcolormesh` 绘制语谱图。`t_stft` 是 X 轴，`f` 是 Y 轴。
      * `np.abs(Zxx)` 计算了复数结果的幅度。通常我们会对幅度取对数 `np.log(np.abs(Zxx))` 或直接使用分贝 `20 * np.log10(np.abs(Zxx))`，这样可以更好地观察到能量较弱的频率成分。这里为了简化，只取了绝对值。
      * `shading='gouraud'` 使颜色过渡更平滑。
      * `cmap='viridis'` 是一种常见的、对色盲友好的颜色映射方案。

#### 结果分析

运行上述代码，你会得到如下图像：

从语谱图中，我们可以清晰地解读出信号的频率结构如何随时间变化：

  * **0 到 2.5 秒**: 图像中有一条清晰的斜线，能量（亮色部分）的频率从大约50Hz平滑地增加到200Hz，完美地捕捉了线性调频信号的特性。
  * **2.5 到 5 秒**: 图像中有两条水平的亮线，分别位于300Hz和400Hz的位置，准确地显示了后半段信号由两个固定频率的正弦波构成。

这证明了 STFT 成功地提供了信号的时间-频率联合表示，这是标准傅里叶变换无法做到的。

### 总结

  * **用途**: STFT 是分析非平稳信号（频率随时间变化的信号）的强大工具。
  * **原理**: 将信号切片、加窗，并对每一片做FFT，最终得到时间-频率图（语谱图）。
  * **核心权衡**: **窗长**决定了时间分辨率和频率分辨率之间的权衡。长窗提供更好的频率分辨率，短窗提供更好的时间分辨率。
  * **Python实现**: `scipy.signal.stft` 是一个高效且易于使用的标准库函数，可以轻松完成STFT分析和可视化。

  [时频分析之STFT：短时傅里叶变换的原理与代码实现（非调用Matlab API）-CSDN博客](https://blog.csdn.net/frostime/article/details/106816373)


