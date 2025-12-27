# Fashion-MNIST Image Classification: SVM vs CNN

这是一个基于 Python 的图像分类项目，旨在通过 **Support Vector Machine (SVM)** 和 **Convolutional Neural Network (CNN)** 两种方法，对 [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集进行分类并对比性能。

*仅作为licy ucas2025秋季学期计算机视觉课程作业*
**START** git clone https://github.com/soleillevant0125/ucas2025cvhomework_licy.git
##  项目结构

```text
.
├── data_loader.py    # 数据加载模块 (解析本地 .gz 压缩包 & 预处理)
├── model_cnn.py      # CNN 模型定义 (PyTorch 实现 LeNet-5 风格)
├── train_svm.py      # SVM 训练与评估逻辑 (sklearn)
├── train_cnn.py      # CNN 训练循环与评估逻辑
├── main.py           # 主程序入口
├── requirements.txt  # 依赖库列表
└── README.md         # 项目说明文档
```

##  环境要求

请确保安装 Python 3.8+，并安装以下依赖：

```
pip install -r requirements.txt
```

主要库：
*   `torch` & `torchvision` (深度学习)
*   `scikit-learn` (SVM)
*   `numpy` (数据处理)

##  数据集准备

由于本项目设计为读取本地原始文件，请按照以下步骤准备数据：

1.  **下载数据**：
    前往 [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)，clone 仓库或下载 `data/fashion` 下的 4 个 `.gz` 文件。

2.  **存放位置**：
    建议将数据放在项目根目录下，例如：`./data/fashion/`。
    确保该目录下包含：
    *   `train-images-idx3-ubyte.gz`
    *   `train-labels-idx1-ubyte.gz`
    *   `t10k-images-idx3-ubyte.gz`
    *   `t10k-labels-idx1-ubyte.gz`

3.  **配置路径**：
    打开 `main.py`，修改 `LOCAL_DATA_PATH` 变量指向你的数据目录：

    ```python
    # main.py
    LOCAL_DATA_PATH = './data/fashion'  # 修改为你实际的绝对或相对路径
    ```

##  运行方法

配置好数据路径后，运行：

```
python main.py
```

## 📈 实验结果 

程序将依次运行 SVM 和 CNN。

| 方法 | 核心/结构 | Accuracy | 时间 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | Linear Kernel | 84.24% | 184.76s | 线性核，速度较快 |
| **SVM** | RBF Kernel | 87.90% | 230.88s | 高斯核，非线性能力更强 |
| **CNN** | LeNet-5 Like | **89.16%** | 7m10s | 2层卷积+全连接，效果最佳 |

*注：SVM 部分默认仅使用了全量训练数据。如需调整训练数据量，在 `main.py` 中 `limit` 参数。*

## 📝 实现细节

1.  **数据预处理**：
    *   **SVM**: 读取数据 -> Flatten 拉平为 (N, 784) -> 归一化 (/255.0)。
    *   **CNN**: 读取数据 -> Reshape 为 (N, 1, 28, 28) -> 归一化 (/255.0) -> 转换为 Tensor。

2.  **CNN 模型**：
    *   自定义 `SimpleCNN` 类，包含卷积层、ReLU 激活函数、最大池化层和全连接层。
    *   手动实现了完整的 Training Loop（前向传播、Loss计算、反向传播、参数更新）。
