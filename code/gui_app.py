"""
MNIST 手写数字识别 — PySide6 上位机
用户在左侧黑色画板上鼠标手写数字，点击"识别"按钮后，
模型预测结果及概率分布显示在右侧。
"""
import sys
import os
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QImage

# 使用 ONNX Runtime 进行推理（避免 PySide6 与 TensorFlow DLL 冲突）
try:
    import onnxruntime as ort
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("警告: 未安装 onnxruntime，请运行 pip install onnxruntime")


class DrawPad(QWidget):
    """28×28 像素手写画板 — 每个像素放大 14 倍显示，便于鼠标绘制"""

    PIXEL_SCALE = 14
    GRID_SIZE = 28
    DISPLAY_SIZE = GRID_SIZE * PIXEL_SCALE  # 392

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.setMouseTracking(True)

        self.pixels = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.drawing = False
        self.last_grid = None

    def paintEvent(self, event):
        painter = QPainter(self)
        h, w = self.pixels.shape
        qimg = QImage(self.pixels.tobytes(), w, h, w, QImage.Format_Grayscale8)
        painter.drawImage(0, 0, qimg.scaled(self.DISPLAY_SIZE, self.DISPLAY_SIZE))

        # 绘制浅灰网格线
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        for i in range(1, self.GRID_SIZE):
            pos = i * self.PIXEL_SCALE
            painter.drawLine(pos, 0, pos, self.DISPLAY_SIZE)
            painter.drawLine(0, pos, self.DISPLAY_SIZE, pos)
        painter.end()

    def _screen_to_grid(self, x, y):
        gx = int(x) // self.PIXEL_SCALE
        gy = int(y) // self.PIXEL_SCALE
        return max(0, min(self.GRID_SIZE - 1, gx)), max(0, min(self.GRID_SIZE - 1, gy))

    def _paint_pixel(self, gx, gy, value=255):
        """绘制单个像素，value 越大越亮"""
        if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
            self.pixels[gy, gx] = max(self.pixels[gy, gx], value)

    def _draw_brush(self, gx, gy):
        """纤细十字笔刷：中心 255 + 四邻 160"""
        self._paint_pixel(gx, gy, 255)
        self._paint_pixel(gx, gy - 1, 160)
        self._paint_pixel(gx, gy + 1, 160)
        self._paint_pixel(gx - 1, gy, 160)
        self._paint_pixel(gx + 1, gy, 160)

    def _draw_line(self, x0, y0, x1, y1):
        """Bresenham 画线法：在两点间连续填充像素"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self._draw_brush(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            gx, gy = self._screen_to_grid(event.position().x(), event.position().y())
            self.last_grid = (gx, gy)
            self._draw_brush(gx, gy)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.LeftButton:
            gx, gy = self._screen_to_grid(event.position().x(), event.position().y())
            if self.last_grid is not None:
                self._draw_line(self.last_grid[0], self.last_grid[1], gx, gy)
            self.last_grid = (gx, gy)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_grid = None

    def clear(self):
        self.pixels.fill(0)
        self.update()

    def get_normalized_image(self):
        return (self.pixels.astype(np.float32) / 255.0).reshape(self.GRID_SIZE, self.GRID_SIZE, 1)


class ProbabilityBarChart(QWidget):
    """概率柱状图：横向展示 0-9 每个数字的预测概率"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.probabilities = np.zeros(10)

    def set_probabilities(self, probs):
        """设置 0-9 的概率数组"""
        self.probabilities = np.array(probs).flatten()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width() - 50
        bar_h = 24
        gap = 6
        total_h = (bar_h + gap) * 10 - gap
        start_y = (self.height() - total_h) // 2

        max_prob = max(np.max(self.probabilities), 0.01)

        for i in range(10):
            prob = self.probabilities[i]
            y = start_y + i * (bar_h + gap)
            bar_w = int((prob / max_prob) * w) if max_prob > 0 else 0

            # 标签
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.drawText(0, y, 25, bar_h, Qt.AlignRight | Qt.AlignVCenter, str(i))

            # 柱状条（高亮预测最高的那个）
            if prob == np.max(self.probabilities) and prob > 0:
                color = QColor(0, 200, 100)  # 绿色
            else:
                color = QColor(80, 150, 220)  # 蓝色
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(30, y, bar_w, bar_h, 4, 4)

            # 概率数值
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 9))
            if bar_w > 50:
                painter.drawText(35, y, bar_w - 5, bar_h, Qt.AlignVCenter, f"{prob:.2%}")
            else:
                painter.drawText(bar_w + 35, y, 60, bar_h, Qt.AlignLeft | Qt.AlignVCenter, f"{prob:.1%}")

        painter.end()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST 手写数字识别")
        self.setMinimumSize(850, 550)

        # 状态变量
        self.session = None  # ONNX Runtime 推理会话
        self.input_name = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.onnx")
        self.fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.h5")

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        """构建界面"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ========== 左侧：画板区域 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.draw_pad = DrawPad()
        left_layout.addWidget(self.draw_pad, alignment=Qt.AlignCenter)

        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清除画板")
        self.btn_clear.setMinimumHeight(40)
        self.btn_clear.setStyleSheet("font-size: 14px; padding: 5px 20px;")
        self.btn_clear.clicked.connect(self.draw_pad.clear)

        self.btn_predict = QPushButton("识别")
        self.btn_predict.setMinimumHeight(40)
        self.btn_predict.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px 30px;"
            "background-color: #4CAF50; color: white; border-radius: 6px;"
        )
        self.btn_predict.clicked.connect(self.on_predict)

        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_predict)
        left_layout.addLayout(btn_layout)

        layout.addWidget(left_widget)

        # ========== 右侧：结果展示 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 8, 10, 8)

        # 预测结果大字
        result_title = QLabel("预测结果")
        result_title.setStyleSheet("font-size: 16px; color: #999;")
        result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_title)

        self.label_result = QLabel("?")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("font-size: 120px; font-weight: bold; color: #4CAF50;")
        right_layout.addWidget(self.label_result)

        self.label_confidence = QLabel("置信度: --")
        self.label_confidence.setAlignment(Qt.AlignCenter)
        self.label_confidence.setStyleSheet("font-size: 18px; color: #aaa;")
        right_layout.addWidget(self.label_confidence)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        right_layout.addWidget(sep)

        # 概率分布标题
        prob_title = QLabel("各类别概率分布")
        prob_title.setStyleSheet("font-size: 14px; color: #888;")
        prob_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(prob_title)

        # 概率柱状图
        self.bar_chart = ProbabilityBarChart()
        right_layout.addWidget(self.bar_chart)

        right_layout.addStretch()
        layout.addWidget(right_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("正在加载模型...")

    def _ensure_onnx_model(self):
        """若不存在 .onnx 但存在 .h5，自动转换为 ONNX 格式"""
        if os.path.exists(self.model_path):
            return True

        if not os.path.exists(self.fallback_path):
            return False

        self.status_bar.showMessage("正在将 .h5 模型转换为 ONNX 格式...")
        try:
            import tensorflow as tf
            import tf2onnx

            model = tf.keras.models.load_model(self.fallback_path)
            model.output_names = ['output']
            spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input'),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            with open(self.model_path, 'wb') as f:
                f.write(model_proto.SerializeToString())
            self.status_bar.showMessage("ONNX 模型转换成功！")
            return True
        except Exception as e:
            self.status_bar.showMessage(f"模型转换失败: {str(e)}")
            self.label_result.setText("!")
            return False

    def _load_model(self):
        """加载 ONNX 模型（自动从 .h5 转换）"""
        if not MODEL_AVAILABLE:
            self.status_bar.showMessage("错误: onnxruntime 未安装，请运行 pip install onnxruntime")
            self.label_result.setText("!")
            return

        if not self._ensure_onnx_model():
            if os.path.exists(self.fallback_path):
                self.status_bar.showMessage("ONNX 转换失败，请检查 TensorFlow 和 tf2onnx 是否安装")
            else:
                self.status_bar.showMessage("错误: 未找到模型文件，请先运行训练笔记本生成 mnist_cnn.h5")
            return

        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.status_bar.showMessage(f"模型已加载: {os.path.basename(self.model_path)} | 等待手写输入...")
        except Exception as e:
            self.status_bar.showMessage(f"模型加载失败: {str(e)}")
            self.label_result.setText("!")

    def on_predict(self):
        """点击识别按钮"""
        if self.session is None:
            self.status_bar.showMessage("模型未加载，无法识别")
            return

        # 1. 从画板获取归一化图像
        img_array = self.draw_pad.get_normalized_image()

        # 2. 添加 batch 维度 (1, 28, 28, 1)
        input_batch = np.expand_dims(img_array, axis=0).astype(np.float32)

        # 3. ONNX Runtime 推理
        predictions = self.session.run(None, {self.input_name: input_batch})[0]
        pred_digit = np.argmax(predictions)
        pred_conf = np.max(predictions)

        # 4. 更新显示
        self.label_result.setText(str(pred_digit))
        self.label_confidence.setText(f"置信度: {pred_conf:.2%}")
        self.bar_chart.set_probabilities(predictions[0])
        self.status_bar.showMessage(f"预测完成 — 结果: {pred_digit}, 置信度: {pred_conf:.2%}")


def main():
    """入口函数"""
    app = QApplication(sys.argv)
    # 设置全局样式
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
