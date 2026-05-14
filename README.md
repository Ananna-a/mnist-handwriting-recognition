# MNIST 手写数字识别 — 神经网络多分类

基于 TensorFlow/Keras 构建 CNN 卷积神经网络，实现对手写数字 0-9 的分类识别。提供 PySide6 手写画板 GUI 上位机，用户可鼠标手写数字并实时获得预测结果和置信度可视化。

---

## 项目结构

```
神经网络回归预测与分类/
├── code/
│   ├── model_training.ipynb     # Jupyter Notebook：模型训练全流程（12节分步教学）
│   ├── gui_app.py               # PySide6 上位机：手写画板 + ONNX实时识别
│   ├── mnist_cnn.h5             # 训练后保存的 Keras 模型（运行 Notebook 生成）
│   └── mnist_cnn.onnx           # ONNX 模型（自动转换，供 GUI 推理使用）
├── mnist/
│   └── mnist.pkl.gz             # MNIST 数据集（70,000 张 28×28 灰度手写数字）
└── requirements.txt             # Python 依赖清单
```

---

## 环境搭建

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 验证安装**

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

按顺序逐单元格执行，完成 12 个章节：
1. 环境准备
2. 数据加载与探索（值域检查、标签分布）
3. 数据预处理（reshape、合并重分）
4. 数据可视化（样本网格展示）
5. CNN 原理详解（卷积→池化→全连接→Dropout→Softmax）
6. 模型搭建与编译（~225k 参数）
7. 模型训练（Loss/Accuracy 曲线）
8. 模型评估（混淆矩阵 + 分类报告）
9. 预测验证与错误分析
10. 保存模型（.h5 + .onnx 自动转换）
11. 调参实验区（含数据增强）
12. 思考题（附参考答案）

> ⚠️ **注意**：该数据集已预归一化到 [0, 1]，无需再做 `/255` 操作。

### 第 2 步：启动 GUI 上位机

```bash
cd code
python gui_app.py
```

启动后：
1. 在左侧 400×400 黑色画板上用鼠标手写数字
2. 点击 **"识别"** 按钮
3. 右侧显示：预测结果 + 置信度 + 28×28 预处理预览 + 概率柱状图
4. 点击 **"清除画板"** 重置画布

> GUI 会自动加载 `mnist_cnn.onnx`（若不存在则从 `.h5` 自动转换）。

---

## 模型架构

```
Conv2D(32, 3×3, ReLU) → MaxPool2D(2×2) →
Conv2D(64, 3×3, ReLU) → MaxPool2D(2×2) →
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

| 层 | 类型 | 输出形状 | 参数量 | 作用 |
|----|------|----------|--------|------|
| 1 | Conv2D | 26×26×32 | 320 | 提取低级特征（边缘、纹理）|
| 2 | MaxPool2D | 13×13×32 | 0 | 降维 |
| 3 | Conv2D | 11×11×64 | 18,496 | 组合低级特征为高级形状 |
| 4 | MaxPool2D | 5×5×64 | 0 | 再次降维 |
| 5 | Flatten | 1600 | 0 | 展平 |
| 6 | Dense | 128 | 204,928 | 综合所有特征 |
| 7 | Dropout(0.5) | 128 | 0 | 防过拟合 |
| 8 | Dense(Softmax) | 10 | 1,290 | 输出 10 类概率 |
| **合计** | | | **225,034** | |

- **优化器**: Adam (lr=0.001)
- **损失函数**: Sparse Categorical Crossentropy
- **训练参数**: epochs=10, batch_size=64
- **目标准确率**: 测试集 > 97%（加数据增强可达 99%+）

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 深度学习框架 | TensorFlow 2.x / Keras | 模型构建与训练 |
| 推理引擎 | ONNX Runtime | GUI 推理（避免 PySide6 + TF DLL 冲突）|
| 数值计算 | NumPy | 数据处理 |
| GUI 框架 | PySide6 | 手写识别上位机 |
| 数据可视化 | Matplotlib + Seaborn | 训练曲线、混淆矩阵 |
| 图像处理 | Pillow | 画布预处理 |
| 评估工具 | scikit-learn | 分类报告、混淆矩阵 |
| 开发环境 | Jupyter Notebook | 交互式分步教学 |

---

## GUI 预处理管线

```
400×400 画板绘制 → 中心质心对齐 → 边界裁剪(15%留白)
→ 保持比例缩至 20×20 → 居中嵌入 28×28 → 高斯模糊(σ=0.55)
→ 归一化 [0, 1] → ONNX 推理 → 输出 0-9 概率
```

---

## 调参建议

| 参数 | 默认值 | 建议尝试 | 预期影响 |
|------|--------|----------|----------|
| 学习率 | 0.001 | 0.0001 / 0.01 | 小→慢但稳；大→快但震荡 |
| 卷积核数 | (32, 64) | (16,32) / (64,128) | 多→强但慢 |
| Dropout | 0.5 | 0.3 / 0.7 | 大→正则化强 |
| FC 神经元 | 128 | 64 / 256 | 多→容量大 |
| Epochs | 10 | 15 / 20 | 多→可能过拟合 |
| **数据增强** | 无 | rotation/zoom/shift | ⭐ 最有效，可提升至 99%+ |

---

## 常见问题

**Q: 运行 Notebook 时出现 `ModuleNotFoundError`？**
A: 执行 `pip install -r requirements.txt`。

**Q: GUI 启动报错 "未找到模型文件"？**
A: 先运行 Notebook 完整训练并保存模型，再启动 GUI。

**Q: 识别不准？**
A: 可能原因：(1) 预处理管线未匹配 → 检查 28×28 预览； (2) 模型未加数据增强 → Notebook 第 11 节重训。 当前准确率约 97%，加数据增强可达 99%。
