"""
MNIST CNN 训练脚本 — 与 model_training.ipynb 逻辑一致
运行此脚本生成 mnist_cnn.h5 模型文件
"""
import numpy as np
import pickle
import gzip
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TensorFlow 日志输出

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("MNIST CNN 模型训练")
print("=" * 60)

# ---- 1. 加载数据 ----
print("\n[1/5] 加载 MNIST 数据集...")
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'mnist', 'mnist.pkl.gz')
with gzip.open(data_path, 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

X_train_raw, y_train_raw = training_data
X_val_raw, y_val_raw = validation_data
X_test_raw, y_test_raw = test_data

# ---- 2. 预处理 ----
print("[2/5] 数据预处理...")
X_train_raw = X_train_raw.reshape(-1, 28, 28, 1).astype('float32')
X_val_raw   = X_val_raw.reshape(-1, 28, 28, 1).astype('float32')
X_test_raw  = X_test_raw.reshape(-1, 28, 28, 1).astype('float32')

y_train_raw = np.array(y_train_raw)
y_val_raw   = np.array(y_val_raw)
y_test_raw  = np.array(y_test_raw)

# 合并训练+验证，重新 9:1 划分（与原验证集分离，确保测试集完全不参与训练）
X_all = np.concatenate([X_train_raw, X_val_raw], axis=0)
y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)
X_test, y_test = X_test_raw, y_test_raw

print(f"  训练集: {X_train.shape[0]:,} 样本")
print(f"  验证集: {X_valid.shape[0]:,} 样本")
print(f"  测试集: {X_test.shape[0]:,} 样本")

# ---- 3. 构建模型 ----
print("[3/5] 构建 CNN 模型...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='fc1'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_CNN')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---- 4. 训练 ----
print("\n[4/5] 开始训练 (10 epochs)...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_valid, y_valid),
    verbose=2
)

# ---- 5. 评估与保存 ----
print("\n[5/5] 评估模型...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"  测试集 Loss:     {test_loss:.4f}")
print(f"  测试集 Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"{'='*50}")

# 保存模型
model_path = os.path.join(base_dir, 'mnist_cnn.h5')
model.save(model_path)
print(f"\n模型已保存至: {model_path}")
print(f"文件大小: {os.path.getsize(model_path) / 1024:.1f} KB")
print("\n训练完成！现在可以运行 gui_app.py 启动手写识别上位机。")
