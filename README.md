# MNIST 手写数字识别 — 神经网络多分类

基于 TensorFlow/Keras 构建 CNN 卷积神经网络，实现对手写数字 0-9 的分类识别。提供 PySide6 手写画板 GUI 上位机，用户可鼠标手写数字并实时获得预测结果和置信度可视化。

---

## 项目结构

```
神经网络回归预测与分类/
├── code/
│   ├── model_training.ipynb   # Jupyter Notebook：模型训练全流程（分步教学）
│   ├── gui_app.py             # PySide6 上位机：手写画板 + 实时识别
│   └── mnist_cnn.h5           # 训练后保存的 Keras 模型（运行 Notebook 后生成）
├── mnist/
│   └── mnist.pkl.gz           # MNIST 数据集（70000 张 28×28 灰度手写数字）
├── requirements.txt            # Python 依赖清单
└── README.md                   # 本文件
```

---

## 环境搭建

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows 激活
venv\Scripts\activate

# macOS/Linux 激活
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import PySide6; print(PySide6.__version__)"
```

---

## 使用步骤

### 第 1 步：运行训练笔记本

```bash
cd code
jupyter notebook model_training.ipynb
```

按顺序逐单元格执行，完成以下操作：
1. 加载并预处理 MNIST 数据集
2. 可视化训练样本
3. 搭建 CNN 模型（2 卷积层 + 2 全连接层）
4. 训练模型并观察 Loss/Accuracy 曲线
5. 评估模型（混淆矩阵、分类报告）
6. 分析错误案例
7. 保存模型为 `mnist_cnn.h5`

Notebook 第 10 节提供了调参实验区，可按提示调整超参数观察效果变化。

### 第 2 步：启动 GUI 上位机

```bash
cd code
python gui_app.py
```

启动后：
1. 在左侧黑色画板上用鼠标手写数字
2. 点击 **"识别"** 按钮
3. 右侧显示预测结果和各类别概率柱状图
4. 点击 **"清除画板"** 重置画布

---

## 模型架构

```
Conv2D(32, 3×3, ReLU) → MaxPool2D(2×2) →
Conv2D(64, 3×3, ReLU) → MaxPool2D(2×2) →
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

| 层 | 类型 | 输出形状 | 参数量 |
|----|------|----------|--------|
| 1 | Conv2D | 26×26×32 | 320 |
| 2 | MaxPooling2D | 13×13×32 | 0 |
| 3 | Conv2D | 11×11×64 | 18,496 |
| 4 | MaxPooling2D | 5×5×64 | 0 |
| 5 | Flatten | 1600 | 0 |
| 6 | Dense | 128 | 204,928 |
| 7 | Dropout | 128 | 0 |
| 8 | Dense | 10 | 1,290 |
| **合计** | | | **~225,034** |

- **优化器**: Adam (lr=0.001)
- **损失函数**: Sparse Categorical Crossentropy
- **训练参数**: epochs=10, batch_size=64
- **目标准确率**: 测试集 > 98%

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 深度学习框架 | TensorFlow 2.x / Keras | 模型构建与训练 |
| 数值计算 | NumPy | 数据处理 |
| GUI 框架 | PySide6 | 手写识别上位机 |
| 数据可视化 | Matplotlib + Seaborn | 训练曲线、混淆矩阵 |
| 图像处理 | Pillow | 画布缩放与预处理 |
| 评估工具 | scikit-learn | 分类报告、混淆矩阵 |
| 开发环境 | Jupyter Notebook | 交互式分步教学 |

---

## 运行效果

### 训练曲线
![训练曲线](docs/training_curves.png)

### 混淆矩阵
![混淆矩阵](docs/confusion_matrix.png)

### GUI 界面
![GUI界面](docs/gui_screenshot.png)

---

## 调参建议

在 `model_training.ipynb` 第 10 节尝试调整以下参数：

| 参数 | 默认值 | 建议尝试 |
|------|--------|----------|
| 学习率 | 0.001 | 0.0001, 0.01 |
| 卷积核数量 | (32, 64) | (16, 32), (64, 128) |
| 卷积核大小 | 3×3 | 5×5 |
| Dropout | 0.5 | 0.3, 0.7 |
| FC 神经元数 | 128 | 64, 256 |
| Batch Size | 64 | 32, 128 |
| Epochs | 10 | 15, 20 |
| 激活函数 | ReLU | LeakyReLU, ELU |
| 添加 BN 层 | 无 | BatchNormalization 加在 Conv 后 |

---

## 常见问题

**Q: 运行 Notebook 时出现 `ModuleNotFoundError`？**  
A: 确保已激活虚拟环境并执行了 `pip install -r requirements.txt`。

**Q: GUI 启动报错 "未找到模型文件"？**  
A: 需要先运行 `model_training.ipynb` 完整训练并保存 `mnist_cnn.h5`，再启动 GUI。

**Q: 中文字体显示为方框？**  
A: Windows 通常自带中文字体；macOS/Linux 可能需安装中文字体，或修改 `plt.rcParams['font.sans-serif']`。

**Q: GPU 训练支持？**  
A: 安装 `tensorflow-gpu` 或 `tensorflow-cpu` 对应版本，TensorFlow 2.x 会自动检测 GPU。
