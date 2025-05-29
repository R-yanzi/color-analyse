import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.color_annotator.sam_interface.sam_segmentor import SAMSegmentor
from src.color_annotator.training.train_sam import SAMWrapper

class SegmentationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, model, image, points):
        super().__init__()
        self.model = model
        self.image = image
        self.points = points

    def run(self):
        try:
            # 设置图像
            self.model.set_image(self.image)
            self.progress.emit(50)

            # 执行分割
            mask = self.model.predict_from_points(
                np.array(self.points),
                np.ones(len(self.points), dtype=np.int64)
            )
            self.progress.emit(100)

            self.finished.emit(mask)
        except Exception as e:
            self.error.emit(str(e))

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI分割测试工具")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化变量
        self.image = None
        self.points = []
        self.mask = None
        self.model = None
        self.segmentor = None

        # 创建UI
        self.setup_ui()
        
        # 加载模型
        self.load_model()

    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(200)

        # 按钮
        self.load_image_btn = QPushButton("加载图片")
        self.load_image_btn.clicked.connect(self.load_image)
        
        self.clear_points_btn = QPushButton("清除点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        
        self.run_segmentation_btn = QPushButton("执行分割")
        self.run_segmentation_btn.clicked.connect(self.run_segmentation)
        self.run_segmentation_btn.setEnabled(False)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # 提示标签
        self.hint_label = QLabel("点击图片添加前景点\n按住Shift点击添加背景点")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setWordWrap(True)

        # 添加控件到控制面板
        control_layout.addWidget(self.load_image_btn)
        control_layout.addWidget(self.clear_points_btn)
        control_layout.addWidget(self.run_segmentation_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.hint_label)
        control_layout.addStretch()

        # 图像显示区域
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("border: 1px solid gray")
        self.image_view.mousePressEvent = self.on_image_click

        # 结果显示区域
        self.result_view = QLabel()
        self.result_view.setAlignment(Qt.AlignCenter)
        self.result_view.setStyleSheet("border: 1px solid gray")

        # 添加到主布局
        layout.addWidget(control_panel)
        layout.addWidget(self.image_view, stretch=1)
        layout.addWidget(self.result_view, stretch=1)

    def load_model(self):
        try:
            # 加载SAM基础模型
            self.segmentor = SAMSegmentor(
                model_type="vit_b",
                checkpoint_path=str(project_root / "checkpoints" / "sam_vit_b.pth"),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # 加载训练好的模型
            model_path = project_root / "checkpoints" / "best_sam_model.pth"
            if model_path.exists():
                checkpoint = torch.load(str(model_path))
                self.model = SAMWrapper(self.segmentor.sam)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print("成功加载训练后的模型")
            else:
                print("未找到训练后的模型，使用原始SAM模型")
                self.model = self.segmentor

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise ValueError("无法读取图片")

                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.display_image(self.image_view, self.image)
                
                self.clear_points()
                self.clear_points_btn.setEnabled(True)
                self.run_segmentation_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败：{str(e)}")

    def display_image(self, label, image, points=None, mask=None):
        if image is None:
            return

        h, w = image.shape[:2]
        # 计算缩放比例
        label_size = label.size()
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        scale = min(scale_w, scale_h)

        # 缩放图像
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(image.copy(), (new_w, new_h))

        # 绘制点
        if points:
            for x, y in points:
                cv2.circle(
                    display_img,
                    (int(x * scale), int(y * scale)),
                    5, (0, 255, 0), -1
                )

        # 绘制掩码
        if mask is not None:
            mask_display = cv2.resize(mask.astype(np.uint8), (new_w, new_h))
            display_img[mask_display > 0] = cv2.addWeighted(
                display_img[mask_display > 0], 0.7,
                np.array([0, 255, 0]), 0.3, 0
            )

        # 转换为QPixmap并显示
        h, w = display_img.shape[:2]
        qimg = QImage(display_img.data, w, h, w * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def on_image_click(self, event):
        if self.image is None:
            return

        # 获取点击位置
        label_pos = event.pos()
        label_size = self.image_view.size()
        image_h, image_w = self.image.shape[:2]

        # 计算缩放比例
        scale_w = label_size.width() / image_w
        scale_h = label_size.height() / image_h
        scale = min(scale_w, scale_h)

        # 计算图像在label中的偏移
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)
        offset_x = (label_size.width() - new_w) // 2
        offset_y = (label_size.height() - new_h) // 2

        # 转换点击位置到原始图像坐标
        image_x = int((label_pos.x() - offset_x) / scale)
        image_y = int((label_pos.y() - offset_y) / scale)

        # 检查点击是否在图像范围内
        if 0 <= image_x < image_w and 0 <= image_y < image_h:
            self.points.append((image_x, image_y))
            self.display_image(self.image_view, self.image, self.points, self.mask)

    def clear_points(self):
        self.points = []
        self.mask = None
        if self.image is not None:
            self.display_image(self.image_view, self.image)
        self.result_view.clear()

    def run_segmentation(self):
        if not self.points:
            QMessageBox.warning(self, "警告", "请先在图片上添加点")
            return

        self.progress_bar.setVisible(True)
        self.run_segmentation_btn.setEnabled(False)
        self.clear_points_btn.setEnabled(False)

        # 创建并启动分割线程
        self.seg_thread = SegmentationThread(self.model, self.image, self.points)
        self.seg_thread.progress.connect(self.progress_bar.setValue)
        self.seg_thread.finished.connect(self.on_segmentation_finished)
        self.seg_thread.error.connect(self.on_segmentation_error)
        self.seg_thread.start()

    def on_segmentation_finished(self, mask):
        self.mask = mask
        self.display_image(self.image_view, self.image, self.points, self.mask)
        
        # 显示单独的掩码结果
        mask_display = np.zeros_like(self.image)
        mask_display[mask > 0] = [0, 255, 0]
        self.display_image(self.result_view, mask_display)

        self.progress_bar.setVisible(False)
        self.run_segmentation_btn.setEnabled(True)
        self.clear_points_btn.setEnabled(True)

    def on_segmentation_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"分割失败：{error_msg}")
        self.progress_bar.setVisible(False)
        self.run_segmentation_btn.setEnabled(True)
        self.clear_points_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 