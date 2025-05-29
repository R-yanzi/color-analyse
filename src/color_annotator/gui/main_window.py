import os
import cv2
import numpy as np
import json
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QSlider, QLabel, QTableWidget, QAbstractItemView,
    QTableWidgetItem, QPushButton, QHeaderView, QLineEdit, QMessageBox, QCheckBox, QGridLayout,
    QColorDialog, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from .image_viewer import ImageViewer
import matplotlib.pyplot as plt
from io import BytesIO
from src.color_annotator.utils.color_analyzer import ColorAnalyzer  # 新增：导入颜色分析器

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("畲族服饰图像标定工具")
        # 设置初始窗口大小为屏幕大小的80%
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            int(screen.width() * 0.1),   # 转换为整数
            int(screen.height() * 0.1),  # 转换为整数
            int(screen.width() * 0.8),   # 转换为整数
            int(screen.height() * 0.8)    # 转换为整数
        )

        # viewer区域设置为自适应大小
        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(800, 600)  # 设置最小尺寸
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 控制栏
        open_btn = QPushButton("打开图片")
        open_btn.clicked.connect(self.open_image)
        reset_btn = QPushButton("重置视图")
        reset_btn.clicked.connect(self.viewer.reset_view)
        save_btn = QPushButton("保存数据")
        save_btn.clicked.connect(self.save_annotations_to_json)

        zoom_label = QLabel("缩放：")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.setValue(100)
        self.scale_slider.setFixedWidth(200)
        self.scale_slider.setSingleStep(5)
        self.scale_slider.valueChanged.connect(self.slider_zoom)
        self.scale_label = QLabel("100%")
        self.scale_label.setFixedWidth(60)

        control_bar = QHBoxLayout()
        control_bar.addWidget(save_btn)
        control_bar.addWidget(open_btn)
        control_bar.addWidget(reset_btn)
        control_bar.addWidget(zoom_label)
        control_bar.addWidget(self.scale_slider)
        control_bar.addWidget(self.scale_label)
        control_bar.addStretch()
        control_widget = QWidget()
        control_widget.setLayout(control_bar)

        # 表格区域
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(8)
        self.annotation_table.setHorizontalHeaderLabels([
            "编号", "主色", "R", "G", "B", "占比", "可见", "操作"
        ])
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # 调整列宽
        header = self.annotation_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        column_widths = [60, 80, 60, 60, 60, 80, 70, 150]  # 总宽度620
        for i, width in enumerate(column_widths):
            self.annotation_table.setColumnWidth(i, width)
        
        self.annotation_table.setMinimumWidth(620)  # 设置最小宽度
        self.annotation_table.setMaximumWidth(620)  # 设置最大宽度
        
        # 更新表格样式
        self.annotation_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                gridline-color: #e9ecef;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: transparent;
                color: black;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 13px;
            }
        """)

        # 三图预览区域
        self.annotation_preview_label = QLabel("标定区域合成预览")
        self.segmentation_preview_label = QLabel("分割结果可视化预览")
        self.color_pie_chart_label = QLabel("主色比例图")

        preview_style = """
            QLabel {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px;
            }
        """
        
        for label in [
            self.annotation_preview_label,
            self.segmentation_preview_label,
            self.color_pie_chart_label
        ]:
            label.setMinimumSize(320, 320)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(preview_style)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建预览图组件和布局
        preview_size = 300  # 设置预览图固定尺寸
        self.annotation_preview_label.setFixedSize(preview_size, preview_size)
        self.segmentation_preview_label.setFixedSize(preview_size, preview_size)
        
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.annotation_preview_label)
        preview_layout.addWidget(self.segmentation_preview_label)
        preview_layout.setSpacing(20)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setAlignment(Qt.AlignCenter)
        
        preview_widget = QWidget()
        preview_widget.setLayout(preview_layout)
        preview_widget.setFixedWidth(400)

        # 左侧导航栏
        left_nav_layout = QVBoxLayout()
        nav_title = QLabel("导航栏")
        nav_title.setAlignment(Qt.AlignCenter)
        nav_title.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
        """)
        left_nav_layout.addWidget(nav_title)
        left_nav_layout.addSpacing(10)

        btn_run = QPushButton("执行分割")
        btn_save = QPushButton("保存标定")
        btn_clear = QPushButton("清除标注")
        btn_add = QPushButton("增加掩码")
        btn_erase = QPushButton("擦除掩码")
        eraser_box = QCheckBox("圆形橡皮擦")

        for btn in [btn_run, btn_save, btn_clear, btn_add, btn_erase]:
            btn.setFixedSize(320, 40)  # 加宽按钮
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    padding: 8px;
                    border: 1px solid #2980b9;
                    border-radius: 4px;
                    background-color: #3498db;  /* 更改为蓝色 */
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                    border-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #2475a8;
                    border-color: #2475a8;
                }
            """)
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(btn)
            row.addStretch()
            left_nav_layout.addLayout(row)

        box_row = QHBoxLayout()
        box_row.addStretch()
        box_row.addWidget(eraser_box)
        box_row.addStretch()
        left_nav_layout.addLayout(box_row)
        left_nav_layout.addStretch()

        # 添加预览图到左侧导航栏下方
        left_nav_layout.addWidget(preview_widget)

        left_nav_widget = QWidget()
        left_nav_widget.setLayout(left_nav_layout)
        left_nav_widget.setFixedWidth(400)
        left_nav_widget.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                border-right: 1px solid #34495e;
            }
            QLabel {
                color: white;
                background-color: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px;
                border: 1px solid #3498db;
                border-radius: 4px;
                background-color: #1abc9c;
                color: #2c3e50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a085;
                border-color: #16a085;
                color: #1a252f;
            }
            QPushButton:pressed {
                background-color: #148f77;
                border-color: #148f77;
                color: #1a252f;
            }
            QCheckBox {
                color: white;
                font-size: 13px;
                padding: 5px;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 2px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #1abc9c;
                background-color: #2c3e50;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #1abc9c;
                background-color: #1abc9c;
            }
        """)

        # 右侧布局（表格+饼图）
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.annotation_table)
        
        # 饼图容器
        pie_container = QWidget()
        pie_container.setFixedSize(620, 500)  # 调整饼图容器尺寸，使其更接近正方形
        pie_layout = QVBoxLayout(pie_container)
        pie_layout.addWidget(self.color_pie_chart_label)
        pie_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(pie_container)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(640)

        # 主图区域（让它填充所有剩余空间）
        center_layout = QVBoxLayout()
        center_layout.addWidget(control_widget)
        center_layout.addWidget(self.viewer, stretch=1)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        center_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(left_nav_widget)
        main_layout.addWidget(center_widget)
        main_layout.addWidget(right_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 设置整体样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

        # 信号绑定
        self.viewer.scaleChanged.connect(self.update_scale_ui)
        self.viewer.annotationAdded.connect(self.cache_annotation)
        self.viewer.segmentationOverlayReady.connect(self.show_segmentation_preview)  # 添加分割预览信号连接
        btn_run.clicked.connect(self.run_segmentation)
        btn_save.clicked.connect(self.save_annotation)
        btn_clear.clicked.connect(self.viewer.clear_annotations)
        btn_add.clicked.connect(self.viewer.set_add_mode)
        btn_erase.clicked.connect(self.viewer.set_erase_mode)
        eraser_box.stateChanged.connect(self.toggle_eraser_shape)

        self.color_analyzer = ColorAnalyzer()

    def show_segmentation_preview(self, pixmap):
        """显示分割预览"""
        if pixmap is None:
            self.segmentation_preview_label.setText("无分割预览")
            return

        w, h = pixmap.width(), pixmap.height()
        target_size = self.segmentation_preview_label.size()
        max_w, max_h = target_size.width() - 20, target_size.height() - 20  # 留出边距

        if w / h > max_w / max_h:
            scaled = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)
        else:
            scaled = pixmap.scaledToHeight(max_h, Qt.SmoothTransformation)

        self.segmentation_preview_label.setPixmap(scaled)
        self.segmentation_preview_label.setAlignment(Qt.AlignCenter)

    def run_segmentation(self):
        # 清除旧的 pending 掩码状态
        # self.pending_annotation = None
        # self.pending_mask_id = None
        # self.viewer.pending_mask_id = None
        # self.viewer.mask = None  # 防止继续合并老掩码

        self.viewer.run_sam_with_points()

    def cache_annotation(self, color_and_mask):
        """缓存标注信息"""
        color_info, mask_id = color_and_mask
        self.pending_annotation = color_info
        self.pending_mask_id = mask_id
        # self.viewer.set_mask_visibility(mask_id, False)  # 保存前默认隐藏

    def save_annotation(self):
        # 强制从 viewer 获取最新掩码状态
        self.pending_mask_id = self.viewer.pending_mask_id

        if self.pending_mask_id is None:
            print("[提示] 没有待保存的掩码")
            return

        mask_data = self.viewer.masks.get(self.pending_mask_id, {}).get("mask", None)
        if mask_data is None or np.sum(mask_data) == 0:
            print("[提示] 当前掩码为空，无法保存")
            return

        if self.pending_annotation is None:
            color_info = self.viewer.extract_main_color()
            if color_info:
                self.pending_annotation = color_info
            else:
                print("[错误] 无法提取颜色（掩码区域无像素）")
                return

        # 显示前先写入颜色
        self.viewer.masks[self.pending_mask_id]['color'] = self.pending_annotation.rgb
        row = self.find_row_by_mask_id(self.pending_mask_id)

        if row != -1:
            # 是修改：更新已有那一行
            self.update_annotation_row(row, self.pending_annotation, self.pending_mask_id)
        else:
            # 是新增：插入新行
            self.add_annotation_to_table(self.pending_annotation, self.pending_mask_id)

        self.viewer.set_mask_visibility(self.pending_mask_id, True)
        self.pending_annotation = None
        self.pending_mask_id = None
        self.viewer.mask = None
        self.viewer.pending_mask_id = None
        print("[完成] 当前标定保存成功")
        self.update_annotation_preview()
        self.update_color_pie_chart()

    def add_annotation_to_table(self, color_info, mask_id):
        """使用ColorInfo对象添加标注到表格"""
        row = self.annotation_table.rowCount()
        self.annotation_table.insertRow(row)

        r, g, b = color_info.rgb
        percentage = color_info.percentage

        # 编号列（不可编辑）
        id_item = QTableWidgetItem(str(row + 1))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        id_item.setTextAlignment(Qt.AlignCenter)
        self.annotation_table.setItem(row, 0, id_item)

        # 主色块（可点击）- 使用容器实现居中
        color_container = QWidget()
        color_layout = QHBoxLayout(color_container)
        color_layout.setContentsMargins(0, 0, 0, 0)
        
        color_label = ClickableColorLabel()
        color_label.setStyleSheet(f"""
            background-color: rgb({r}, {g}, {b});
            border: 1px solid #dee2e6;
            border-radius: 2px;
        """)
        color_label.setFixedSize(30, 30)  # 正方形颜色块
        
        color_layout.addStretch()
        color_layout.addWidget(color_label)
        color_layout.addStretch()
        
        color_label.clicked.connect(lambda: self.show_color_dialog(row))
        self.annotation_table.setCellWidget(row, 1, color_container)

        # R/G/B 列（可编辑）
        for col, val in zip((2, 3, 4), (r, g, b)):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.annotation_table.setItem(row, col, item)
            
        # 占比列（不可编辑）
        percentage_item = QTableWidgetItem(f"{percentage:.1%}")
        percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
        percentage_item.setTextAlignment(Qt.AlignCenter)
        self.annotation_table.setItem(row, 5, percentage_item)

        # 可见性切换按钮
        visibility_btn = QPushButton()
        visibility_btn.setCheckable(True)
        visibility_btn.setChecked(True)
        visibility_btn.setFixedSize(50, 35)
        visibility_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: #28a745;
                font-family: "Segoe UI Symbol";
                font-size: 20px;
                padding: 0px;
            }
            QPushButton:checked {
                color: #dc3545;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                border-radius: 4px;
            }
        """)
        visibility_btn.setText("●")  # 使用实心圆点表示可见
        visibility_btn.clicked.connect(lambda checked: self.toggle_mask_visibility(row, checked))
        visibility_btn.setProperty("mask_id", mask_id)
        
        visibility_widget = QWidget()
        visibility_layout = QHBoxLayout(visibility_widget)
        visibility_layout.addWidget(visibility_btn)
        visibility_layout.setAlignment(Qt.AlignCenter)
        visibility_layout.setContentsMargins(0, 0, 0, 0)
        self.annotation_table.setCellWidget(row, 6, visibility_widget)

        # 操作按钮
        op_widget = QWidget()
        op_layout = QHBoxLayout()
        op_layout.setContentsMargins(5, 0, 5, 0)
        op_layout.setSpacing(5)

        edit_btn = QPushButton("修改")
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 2px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        edit_btn.clicked.connect(lambda: self.edit_annotation(row, mask_id))

        del_btn = QPushButton("删除")
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 2px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        del_btn.clicked.connect(lambda: self.delete_annotation(mask_id, row))

        op_layout.addWidget(edit_btn)
        op_layout.addWidget(del_btn)
        op_widget.setLayout(op_layout)
        self.annotation_table.setCellWidget(row, 7, op_widget)

        # 设置行高
        self.annotation_table.setRowHeight(row, 40)  # 增加行高

    def delete_annotation(self, mask_id, row):
        """删除标注"""
        reply = QMessageBox.question(
            self,
            "确认删除",
            "是否确定删除该标定？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 从masks字典中删除
                if mask_id in self.viewer.masks:
                    del self.viewer.masks[mask_id]
                
                # 从表格中删除行
                self.annotation_table.removeRow(row)
                
                # 更新剩余行的编号
                for i in range(self.annotation_table.rowCount()):
                    id_item = self.annotation_table.item(i, 0)
                    if id_item:
                        id_item.setText(str(i + 1))
                
                # 清理相关状态
                if self.viewer.pending_mask_id == mask_id:
                    self.viewer.pending_mask_id = None
                    self.viewer.mask = None
                
                # 如果删除后没有任何标注，清空饼图
                if self.annotation_table.rowCount() == 0:
                    self.color_pie_chart_label.clear()
                    self.color_pie_chart_label.setText("无标注数据")
                    self.annotation_preview_label.clear()
                    self.annotation_preview_label.setText("无标注数据")
                else:
                    # 刷新显示
                    self.update_annotation_preview()
                    self.update_color_pie_chart()
                
                # 刷新视图
                self.viewer.repaint()
                
                print(f"[删除] 掩码 {mask_id} 已删除")
            except Exception as e:
                print(f"[错误] 删除失败: {e}")
                QMessageBox.warning(self, "删除失败", f"删除操作出现错误：{str(e)}")
        else:
            print("[取消] 删除操作已取消")

    @staticmethod
    def encode_rle(mask: np.ndarray):
        flat = mask.flatten()
        rle = []
        i = 0
        while i < len(flat):
            if flat[i] == 1:
                start = i
                length = 1
                i += 1
                while i < len(flat) and flat[i] == 1:
                    length += 1
                    i += 1
                rle.append([start, length])
            else:
                i += 1
        return rle

    def update_annotation_preview(self):
        if self.viewer.cv_img is None:
            return

        img = self.viewer.cv_img.copy()
        overlay = np.zeros_like(img, dtype=np.uint8)

        for entry in self.viewer.masks.values():
            if not entry.get("visible", True):
                continue
            mask = entry.get("mask")
            if mask is None or mask.shape != overlay.shape[:2]:
                continue
            color = entry.get("color", (0, 255, 0))
            r, g, b = color
            overlay[mask] = (b, g, r)

        # 叠加时只让掩码部分半透明，其余部分完全显示原图
        alpha = 0.9
        mask_area = (overlay > 0).any(axis=2)
        preview = img.copy()
        preview[mask_area] = cv2.addWeighted(img[mask_area], 1 - alpha, overlay[mask_area], alpha, 0)

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w, _ = preview_rgb.shape
        qimg = QImage(preview_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.annotation_preview_label.setPixmap(
            pixmap.scaled(self.annotation_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_color_pie_chart(self):
        """更新颜色饼图"""
        if self.viewer.cv_img is None:
            self.color_pie_chart_label.setText("无图片")
            return

        # 从表格中收集颜色信息
        colors_data = []
        for row in range(self.annotation_table.rowCount()):
            try:
                # 获取RGB值
                r = int(self.annotation_table.item(row, 2).text() or 0)
                g = int(self.annotation_table.item(row, 3).text() or 0)
                b = int(self.annotation_table.item(row, 4).text() or 0)
                # 获取占比
                percentage_text = self.annotation_table.item(row, 5).text().rstrip('%')
                percentage = float(percentage_text) / 100.0
                
                colors_data.append({
                    'rgb': (r, g, b),
                    'percentage': percentage
                })
            except (ValueError, AttributeError) as e:
                print(f"[警告] 行 {row + 1} 数据无效: {e}")
                continue

        if not colors_data:
            self.color_pie_chart_label.setText("无标注数据")
            return

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制饼图
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))

        # 准备数据
        sizes = [data['percentage'] for data in colors_data]
        colors = [f"#{r:02x}{g:02x}{b:02x}" for data in colors_data for r, g, b in [data['rgb']]]

        # 绘制饼图，不显示文字标签
        wedges, texts = ax.pie(
            sizes,
            colors=colors,
            labels=[''] * len(colors),  # 不显示标签
            autopct=None,  # 不显示百分比
            startangle=90
        )

        # 添加图例，显示颜色和百分比
        legend_labels = [f'{data["percentage"]:.1%}' for data in colors_data]
        legend = ax.legend(
            wedges,
            legend_labels,
            title="颜色占比",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=14,  # 增大图例文字大小
            title_fontsize=16  # 增大图例标题大小
        )

        # 调整图例样式
        legend.get_frame().set_linewidth(2)  # 加粗图例边框
        legend.get_frame().set_edgecolor('black')  # 设置图例边框颜色

        ax.axis('equal')

        # 保存图片，增加边距以显示完整图例
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, pad_inches=0.3)
        buf.seek(0)
        plt.close(fig)

        # 显示图片
        img = QImage.fromData(buf.read())
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            self.color_pie_chart_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.color_pie_chart_label.setPixmap(scaled_pixmap)
        self.color_pie_chart_label.setAlignment(Qt.AlignCenter)

    def toggle_mask_visibility(self, row, checked):
        """切换掩码可见性"""
        try:
            visibility_widget = self.annotation_table.cellWidget(row, 6)
            if not visibility_widget:
                return
                
            visibility_btn = visibility_widget.findChild(QPushButton)
            if not visibility_btn:
                return

            mask_id = visibility_btn.property("mask_id")
            if not mask_id:
                return

            # 从当前行提取 R/G/B 值，添加默认值处理
            r = int(self.annotation_table.item(row, 2).text() or 0)
            g = int(self.annotation_table.item(row, 3).text() or 0)
            b = int(self.annotation_table.item(row, 4).text() or 0)

            if checked:
                visibility_btn.setText("●")  # 实心圆点表示可见
                self.viewer.set_mask_visibility(mask_id, True, (r, g, b))
            else:
                visibility_btn.setText("○")  # 空心圆圈表示不可见
                self.viewer.set_mask_visibility(mask_id, False)

            self.update_annotation_preview()
            self.update_color_pie_chart()

        except Exception as e:
            print(f"[错误] 切换掩码可见性失败: {str(e)}")

    def edit_annotation(self, row, mask_id):
        print(f"[编辑] 开始编辑掩码 {mask_id}")
        if mask_id in self.viewer.masks:
            # 设置该掩码为可编辑状态
            self.viewer.masks[mask_id]['editable'] = True
            self.viewer.masks[mask_id]['visible'] = True
            self.viewer.pending_mask_id = mask_id
            self.viewer.mask = self.viewer.masks[mask_id]['mask']
            self.viewer.update()

    def save_annotations_to_json(self):
        """保存标注数据到JSON文件"""
        if self.viewer.cv_img is None:
            QMessageBox.warning(self, "未加载图像", "请先打开一张图片。")
            return

        try:
            # 检查图像路径
            if not hasattr(self.viewer, 'image_path') or not self.viewer.image_path:
                QMessageBox.warning(self, "保存失败", "无法确定图像路径。")
                return

            # 创建保存路径
            base_name = os.path.splitext(os.path.basename(self.viewer.image_path))[0]
            save_path = os.path.join("annotations", base_name + ".json")
            os.makedirs("annotations", exist_ok=True)

            # 获取相对路径
            abs_img_path = self.viewer.image_path
            proj_root = os.path.abspath(os.getcwd())
            rel_img_path = os.path.relpath(abs_img_path, proj_root)

            # 准备数据
            annotations = []
            for mask_id, entry in self.viewer.masks.items():
                try:
                    if entry is None or 'mask' not in entry:
                        print(f"[警告] 掩码 {mask_id} 数据无效，跳过")
                        continue

                    mask_array = entry["mask"]
                    if mask_array is None or mask_array.size == 0:
                        print(f"[警告] 掩码 {mask_id} 为空，跳过")
                        continue

                    mask_array = mask_array.astype(np.int32)
                    height, width = mask_array.shape
                    rle = self.encode_rle(mask_array)

                    # 确保颜色值是整数列表
                    color = entry.get("color", [0, 255, 0])
                    color = [int(c) for c in color]

                    annotation = {
                        "rle": [[int(s), int(l)] for s, l in rle],
                        "size": [int(height), int(width)],
                        "main_color": color
                    }
                    annotations.append(annotation)
                except Exception as e:
                    print(f"[警告] 处理掩码 {mask_id} 时出错: {str(e)}")
                    continue

            # 如果没有有效的标注，提示用户
            if not annotations:
                QMessageBox.warning(self, "保存失败", "没有有效的标注数据可保存。")
                return

            # 保存数据
            data = {
                "image_path": rel_img_path.replace("\\", "/"),
                "annotations": annotations
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "保存成功", f"标定数据已保存至：\n{save_path}")
            
        except Exception as e:
            error_msg = f"保存失败: {str(e)}"
            print(f"[错误] {error_msg}")
            QMessageBox.critical(self, "保存失败", error_msg)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./read_images", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        if file_path:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext in ['.tif', '.tiff']:
                from PIL import Image
                img_pil = Image.open(file_path).convert("RGB")  # 转换为 RGB 三通道
                img = np.array(img_pil)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 保持与原有代码一致的 BGR 格式
            else:
                img = cv2.imread(file_path)

            if img is None:
                QMessageBox.warning(self, "错误", "无法加载图片文件")
                return

            self.viewer.set_image(img)
            self.viewer.image_path = file_path

            # === 自动加载对应 JSON 文件 ===
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join("annotations", base_name + ".json")
            
            try:
                masks = self.viewer.load_masks_from_json(json_path)
                if masks:
                    # === 同步表格显示 ===
                    self.annotation_table.setRowCount(0)
                    for mask_id, entry in masks.items():
                        # 从 ColorInfo 或者 RGB 元组创建颜色信息
                        color = entry.get("color", (0, 255, 0))
                        if isinstance(color, (list, tuple)):
                            from src.color_annotator.utils.color_analyzer import ColorInfo
                            # 计算掩码占比
                            mask = entry.get("mask", None)
                            if mask is not None:
                                percentage = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                            else:
                                percentage = 0
                            color_info = ColorInfo(rgb=tuple(color), percentage=percentage)
                        else:
                            color_info = color

                        # 添加到表格
                        self.add_annotation_to_table(color_info, mask_id)

                        # 设置掩码显示状态（默认显示）
                        visible = entry.get("visible", True)
                        self.viewer.set_mask_visibility(mask_id, visible)

                        # 找到对应行，设置显示列为 ✔ / ✖
                        row = self.find_row_by_mask_id(mask_id)
                        if row != -1:
                            show_item = self.annotation_table.item(row, 6)
                            if show_item:
                                show_item.setText("✔" if visible else "✖")

                    # 更新预览
                    self.update_annotation_preview()
                    self.update_color_pie_chart()
                else:
                    # 清空显示
                    self.annotation_table.setRowCount(0)
                    self.annotation_preview_label.clear()
                    self.segmentation_preview_label.clear()
                    self.color_pie_chart_label.clear()
                    self.annotation_preview_label.setText("无标注数据")
                    self.color_pie_chart_label.setText("无标注数据")

            except Exception as e:
                print(f"[错误] 加载标注数据失败: {str(e)}")
                QMessageBox.warning(self, "加载失败", f"加载标注数据时出错：{str(e)}")
                # 出错时也清空显示
                self.annotation_table.setRowCount(0)
                self.annotation_preview_label.clear()
                self.segmentation_preview_label.clear()
                self.color_pie_chart_label.clear()

            # 更新缩放UI
            self.update_scale_ui()
            slider_val = int(self.viewer.scale / self.viewer.base_scale * 100)
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(slider_val)
            self.scale_slider.blockSignals(False)

    def handle_color_change(self, item):
        row = item.row()
        col = item.column()

        # 仅处理 R/G/B 列（2/3/4）
        if col not in (2, 3, 4):
            return

        try:
            r = int(self.annotation_table.item(row, 2).text())
            g = int(self.annotation_table.item(row, 3).text())
            b = int(self.annotation_table.item(row, 4).text())
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
        except:
            return  # 输入非法，跳过

        # 更新主色块显示
        color_label = self.annotation_table.cellWidget(row, 1)
        if isinstance(color_label, QLabel):
            color_label.setStyleSheet(f"""
                background-color: rgb({r}, {g}, {b});
                border: 1px solid gray;
                margin-left: 5px;
                margin-right: 5px;
            """)

        # 更新掩码颜色并刷新视图
        show_item = self.annotation_table.item(row, 6)
        if show_item:
            mask_id = show_item.data(Qt.UserRole)
            if mask_id and mask_id in self.viewer.masks:
                self.viewer.masks[mask_id]['color'] = (r, g, b)
                self.viewer.update()

    def slider_zoom(self, value):
        scale = self.viewer.base_scale * (value / 100.0)
        self.viewer.set_scale(scale)
        self.update_scale_ui()

    def update_annotation_row(self, row, color_info, mask_id):
        """更新已有行的颜色信息"""
        r, g, b = color_info.rgb
        percentage = color_info.percentage

        # 更新RGB值
        self.annotation_table.setItem(row, 2, QTableWidgetItem(str(r)))
        self.annotation_table.setItem(row, 3, QTableWidgetItem(str(g)))
        self.annotation_table.setItem(row, 4, QTableWidgetItem(str(b)))

        # 更新占比
        percentage_item = QTableWidgetItem(f"{percentage:.1%}")
        percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
        percentage_item.setTextAlignment(Qt.AlignCenter)
        self.annotation_table.setItem(row, 5, percentage_item)

        # 更新颜色块
        color_label = QLabel()
        color_label.setStyleSheet(f"""
            background-color: rgb({r},{g},{b});
            border: 1px solid gray;
            margin-left: 5px;
            margin-right: 5px;
        """)
        color_label.setFixedSize(40, 20)
        color_label.setAlignment(Qt.AlignCenter)
        self.annotation_table.setCellWidget(row, 1, color_label)

    def find_row_by_mask_id(self, mask_id):
        for row in range(self.annotation_table.rowCount()):
            show_item = self.annotation_table.item(row, 6)
            if show_item:
                if show_item.data(Qt.UserRole) == mask_id:
                    return row
        return -1

    def update_scale_ui(self):
        slider_val = int(self.viewer.scale / self.viewer.base_scale * 100)
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(slider_val)
        self.scale_slider.blockSignals(False)
        self.scale_label.setText(f"{slider_val}%")

    def toggle_eraser_shape(self, state):
        self.viewer.eraser_shape_circle = (state == Qt.Checked)
        self.viewer.update()

    def show_color_dialog(self, row):
        """显示颜色选择对话框"""
        try:
            # 获取当前RGB值
            r = int(self.annotation_table.item(row, 2).text())
            g = int(self.annotation_table.item(row, 3).text())
            b = int(self.annotation_table.item(row, 4).text())
            
            # 创建颜色对话框
            color_dialog = QColorDialog(self)
            color_dialog.setCurrentColor(QColor(r, g, b))
            color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
            color_dialog.setWindowTitle("选择颜色")
            
            if color_dialog.exec_():
                # 获取新选择的颜色
                new_color = color_dialog.currentColor()
                new_r, new_g, new_b = new_color.red(), new_color.green(), new_color.blue()
                
                # 更新RGB值
                self.annotation_table.item(row, 2).setText(str(new_r))
                self.annotation_table.item(row, 3).setText(str(new_g))
                self.annotation_table.item(row, 4).setText(str(new_b))
                
                # 更新颜色块显示
                color_label = self.annotation_table.cellWidget(row, 1)
                if isinstance(color_label, QLabel):
                    color_label.setStyleSheet(f"""
                        background-color: rgb({new_r}, {new_g}, {new_b});
                        border: 1px solid gray;
                        margin-left: 5px;
                        margin-right: 5px;
                    """)
                
                # 更新掩码颜色
                show_item = self.annotation_table.item(row, 6)
                if show_item:
                    mask_id = show_item.data(Qt.UserRole)
                    if mask_id and mask_id in self.viewer.masks:
                        self.viewer.masks[mask_id]['color'] = (new_r, new_g, new_b)
                        self.viewer.update()
                
                # 更新颜色饼图
                self.update_color_pie_chart()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"修改颜色时出错：{str(e)}")

# 添加可点击的颜色标签类
class ClickableColorLabel(QLabel):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针为手型
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

