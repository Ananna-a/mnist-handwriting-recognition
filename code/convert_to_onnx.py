"""将训练好的 .h5 模型转换为 ONNX 格式"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tf2onnx

# 加载模型
model = tf.keras.models.load_model('mnist_cnn.h5')
model.output_names = ['output']

# 定义输入规格
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input'),)

# 转换
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# 保存
output_path = 'mnist_cnn.onnx'
with open(output_path, 'wb') as f:
    f.write(model_proto.SerializeToString())

size_kb = os.path.getsize(output_path) / 1024
print(f"ONNX 模型已保存: {output_path} ({size_kb:.1f} KB)")
