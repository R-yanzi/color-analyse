import os
import cv2
import numpy as np
import json
from PyQt5.QtGui import QColor, QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QSlider, QLabel, QTableWidget, QAbstractItemView,
    QTableWidgetItem, QPushButton, QHeaderView, QLineEdit, QMessageBox, QCheckBox, QGridLayout,
    QColorDialog, QApplication, QSizePolicy, QDockWidget, QShortcut
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from .image_viewer import ImageViewer
import matplotlib.pyplot as plt
from io import BytesIO
from src.color_annotator.utils.color_analyzer import ColorAnalyzer  # 新增：导入颜色分析器
from .ai_segmentation import AISegmentationWidget
from pathlib import Path
import time
from PyQt5.QtGui import QKeySequence


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("畲族服饰图像标定工具")
        # 设置初始窗口大小为屏幕大小的80%
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            int(screen.width() * 0.1),  # 转换为整数
            int(screen.height() * 0.1),  # 转换为整数
            int(screen.width() * 0.8),  # 转换为整数
            int(screen.height() * 0.8)  # 转换为整数
        )

        # 初始化保存标定相关的属性
        self.pending_annotation = None
        self.pending_mask_id = None

        # 放大镜启用状态
        self.magnifier_enabled = True

        # viewer区域设置为自适应大小
        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(800, 600)  # 设置最小尺寸
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建浮动放大镜窗口 - 使用相对尺寸
        self.magnifier_window = QLabel(self.viewer)
        # 初始大小设置，后面会根据窗口大小动态调整
        self.update_magnifier_size()
        self.magnifier_window.setAlignment(Qt.AlignCenter)
        self.magnifier_window.setStyleSheet("""
            background-color: rgba(0, 0, 0, 140);
            color: white;
            border: 1px solid white;
            border-radius: 5px;
            padding: 1px;
        """)
        self.magnifier_window.hide()  # 初始时隐藏

        # 创建放大镜更新定时器 - 降低检查频率减少资源消耗
        self.magnifier_timer = QTimer(self)
        self.magnifier_timer.timeout.connect(self.check_magnifier_status)
        self.magnifier_timer.start(100)  # 100毫秒检查一次(原为50ms)

        # 控制栏
        open_btn = QPushButton("打开图片 (Ctrl+O)")
        open_btn.clicked.connect(self.open_image)
        open_btn.setShortcut("Ctrl+O")

        reset_btn = QPushButton("重置视图 (R)")
        reset_btn.clicked.connect(self.viewer.reset_view)
        reset_btn.setShortcut("R")

        save_btn = QPushButton("保存数据 (F2)")
        save_btn.clicked.connect(self.save_annotations_to_json)
        save_btn.setShortcut("F2")

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
        self.annotation_table.setColumnCount(5)  # 从8列减少为5列
        self.annotation_table.setHorizontalHeaderLabels([
            "编号", "主色", "占比", "可见", "操作"  # 删除RGB三列
        ])
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)

        # 调整列宽
        header = self.annotation_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        column_widths = [50, 150, 80, 70, 170]  # 调整列宽分配
        for i, width in enumerate(column_widths):
            self.annotation_table.setColumnWidth(i, width)

        self.annotation_table.setMinimumWidth(520)  # 调整总宽度
        self.annotation_table.setMaximumWidth(520)  # 调整总宽度

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

        # 预览区域
        self.annotation_preview_label = QLabel("标定区域合成预览")
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
            self.color_pie_chart_label
        ]:
            label.setMinimumSize(320, 320)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(preview_style)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建预览图组件和布局
        preview_size = 300  # 设置预览图固定尺寸
        self.annotation_preview_label.setFixedSize(preview_size, preview_size)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.annotation_preview_label)
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

        btn_run = QPushButton("执行分割 (F5)")
        btn_save = QPushButton("保存标定 (Ctrl+S)")
        btn_clear = QPushButton("清除标注 (Ctrl+X)")
        self.btn_add = QPushButton("增加掩码 (A)")  # 保存为实例变量
        self.btn_erase = QPushButton("擦除掩码 (E)")  # 保存为实例变量
        eraser_box = QCheckBox("圆形橡皮擦")
        magnifier_box = QCheckBox("显示放大镜")  # 添加放大镜开关

        # 设置快捷键
        btn_run.setShortcut("F5")
        btn_save.setShortcut("Ctrl+S")
        btn_clear.setShortcut("Ctrl+X")
        # 增加/擦除掩码的快捷键在toggle_add_mode和toggle_erase_mode方法中设置

        # 设置按钮样式
        self.button_style = """
            QPushButton {
                font-size: 16px;
                padding: 8px;
                border: 1px solid #2980b9;
                border-radius: 4px;
                background-color: #3498db;
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
        """

        # 设置激活状态的样式
        self.active_button_style = """
            QPushButton {
                font-size: 16px;
                padding: 8px;
                border: 1px solid #16a085;
                border-radius: 4px;
                background-color: #1abc9c;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a085;
                border-color: #16a085;
            }
        """

        for btn in [btn_run, btn_save, btn_clear, self.btn_add, self.btn_erase]:
            btn.setFixedSize(320, 40)  # 加宽按钮
            btn.setStyleSheet(self.button_style)
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(btn)
            row.addStretch()
            left_nav_layout.addLayout(row)

        # 创建复选框行
        checkbox_row = QHBoxLayout()
        checkbox_row.addStretch()
        checkbox_row.addWidget(eraser_box)
        checkbox_row.addSpacing(20)  # 添加间距
        checkbox_row.addWidget(magnifier_box)
        checkbox_row.addStretch()
        left_nav_layout.addLayout(checkbox_row)

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

        # 添加导航栏控制按钮
        self.toggle_nav_btn = QPushButton("◀")  # 初始状态为隐藏导航栏，显示右箭头
        self.toggle_nav_btn.setFixedSize(30, 60)
        self.toggle_nav_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2c3e50;
                        color: white;
                        border: 1px solid #34495e;
                        border-radius: 3px;
                        font-size: 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #34495e;
                    }
                """)
        self.toggle_nav_btn.clicked.connect(self.toggle_navigation)

        # 将导航栏和控制按钮放入一个容器中
        nav_container = QWidget()
        nav_container_layout = QHBoxLayout(nav_container)
        nav_container_layout.setContentsMargins(0, 0, 0, 0)
        nav_container_layout.setSpacing(0)
        nav_container_layout.addWidget(left_nav_widget)
        nav_container_layout.addWidget(self.toggle_nav_btn)

        # 默认隐藏导航栏
        self.nav_visible = False
        left_nav_widget.hide()
        self.toggle_nav_btn.setText("▶")  # 改为右箭头表示可以展开



        # 右侧布局（表格+饼图）
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.annotation_table)

        # 饼图容器
        pie_container = QWidget()
        pie_container.setFixedSize(520, 480)  # 调整饼图容器尺寸，使其更窄
        pie_layout = QVBoxLayout(pie_container)
        pie_layout.addWidget(self.color_pie_chart_label)
        pie_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(pie_container)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(10, 10, 10, 10)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(540)  # 减小右侧区域宽度

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
        main_layout.setSpacing(0)  # 设置为0以避免间距
        main_layout.addWidget(nav_container)
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
        self.viewer.magnifierUpdated.connect(self.update_magnifier_preview)
        btn_run.clicked.connect(self.run_segmentation)
        btn_save.clicked.connect(self.save_annotation)
        btn_clear.clicked.connect(self.viewer.clear_annotations)
        self.btn_add.clicked.connect(self.toggle_add_mode)
        self.btn_erase.clicked.connect(self.toggle_erase_mode)
        eraser_box.stateChanged.connect(self.toggle_eraser_shape)

        self.color_analyzer = ColorAnalyzer()

        # 创建AI分割工具
        self.setupAISegmentation()
        self.dock_ai = QDockWidget("AI分割", self)
        self.dock_ai.setWidget(self.ai_segmentation)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_ai)

        # 配置快捷键
        self.add_shortcut = QShortcut(QKeySequence("A"), self)
        self.add_shortcut.activated.connect(self.toggle_add_mode)

        self.erase_shortcut = QShortcut(QKeySequence("E"), self)
        self.erase_shortcut.activated.connect(self.toggle_erase_mode)

        # 配置放大镜开关
        magnifier_box.setChecked(True)  # 默认打开放大镜
        magnifier_box.stateChanged.connect(self.toggle_magnifier)

    def toggle_navigation(self):
        """切换导航栏显示/隐藏状态"""
        # 找到导航栏部件
        nav_container = self.toggle_nav_btn.parent()
        left_nav_widget = nav_container.layout().itemAt(0).widget()

        if self.nav_visible:
            left_nav_widget.hide()
            self.toggle_nav_btn.setText("▶")  # 右箭头表示可以展开
            # 调整按钮位置到最左侧
            self.toggle_nav_btn.setParent(nav_container)
            nav_container.layout().removeWidget(self.toggle_nav_btn)
            nav_container.layout().addWidget(self.toggle_nav_btn)
        else:
            left_nav_widget.show()
            self.toggle_nav_btn.setText("◀")  # 左箭头表示可以收起
            # 调整按钮位置到导航栏右侧
            self.toggle_nav_btn.setParent(nav_container)
            nav_container.layout().removeWidget(self.toggle_nav_btn)
            nav_container.layout().insertWidget(1, self.toggle_nav_btn)

        self.nav_visible = not self.nav_visible

    def run_segmentation(self):
        """执行基于点的人工标定分割"""
        try:
            print("[用户操作] 执行基于点的分割")
            if self.viewer.cv_img is None:
                print("[错误] 没有加载图像")
                return

            # 调用viewer的点标注分割方法
            self.viewer.run_sam_with_points()

        except Exception as e:
            import traceback
            print(f"[错误] 执行分割时出错: {str(e)}")
            print(traceback.format_exc())

    def cache_annotation(self, color_and_mask):
        """缓存标注信息"""
        color_info, mask_id = color_and_mask
        self.pending_annotation = color_info
        self.pending_mask_id = mask_id
        # self.viewer.set_mask_visibility(mask_id, False)  # 保存前默认隐藏

    def save_annotation(self):
        """保存当前标定"""
        try:
            # 强制从 viewer 获取最新掩码状态
            self.pending_mask_id = self.viewer.pending_mask_id

            if self.pending_mask_id is None:
                print("[提示] 没有待保存的掩码")
                return

            mask_data = self.viewer.masks.get(self.pending_mask_id, {}).get("mask", None)
            if mask_data is None or np.sum(mask_data) == 0:
                print("[提示] 当前掩码为空，无法保存")
                return

            # 检查是否是在编辑现有标定
            row = self.find_row_by_mask_id(self.pending_mask_id)
            is_editing = row != -1

            if is_editing:
                # 编辑现有标定：保持原有颜色
                print(f"[保存] 更新现有标定 {self.pending_mask_id}")

                # 从ID项获取存储的RGB值 - 修改这部分代码
                id_item = self.annotation_table.item(row, 0)
                if id_item and id_item.data(Qt.UserRole + 1):
                    r, g, b = id_item.data(Qt.UserRole + 1)
                else:
                    # 如果无法获取，使用默认颜色
                    r, g, b = 0, 255, 0
                    print("[警告] 无法获取颜色信息，使用默认颜色")

                self.viewer.masks[self.pending_mask_id]['color'] = (r, g, b)

                # 重新计算所有掩码的占比
                total_pixels = self.viewer.cv_img.shape[0] * self.viewer.cv_img.shape[1]
                visible_masks = {k: v for k, v in self.viewer.masks.items() if v.get('visible', True)}

                # 计算所有可见掩码的总像素数
                total_mask_pixels = sum(np.sum(mask['mask']) for mask in visible_masks.values())

                # 更新当前掩码的占比
                current_mask_pixels = np.sum(mask_data)
                if total_mask_pixels > 0:
                    percentage = current_mask_pixels / total_pixels
                else:
                    percentage = 0

                # 创建颜色信息对象
                from src.color_annotator.utils.color_analyzer import ColorInfo
                color_info = ColorInfo(rgb=(r, g, b), percentage=percentage)

                # 更新行显示
                self.update_annotation_row(row, color_info, self.pending_mask_id)

            else:
                # 新增标定：提取新的颜色
                print("[保存] 创建新标定")
                if not hasattr(self, 'pending_annotation') or self.pending_annotation is None:
                    try:
                        color_info = self.viewer.extract_main_color()
                        if color_info:
                            self.pending_annotation = color_info
                        else:
                            print("[错误] 无法提取颜色（掩码区域无像素）")
                            return
                    except Exception as e:
                        print(f"[错误] 提取颜色时出错: {str(e)}")
                        return

                # 写入颜色并添加新行
                self.viewer.masks[self.pending_mask_id]['color'] = self.pending_annotation.rgb
                self.add_annotation_to_table(self.pending_annotation, self.pending_mask_id)

            # 设置为可见
            self.viewer.set_mask_visibility(self.pending_mask_id, True)

            # 清理状态
            self.pending_annotation = None
            self.viewer.mask = None
            self.viewer.pending_mask_id = None

            # 更新显示
            self.update_annotation_preview()
            self.update_color_pie_chart()

            print("[完成] 当前标定保存成功")

        except Exception as e:
            print(f"[错误] 保存标定时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "保存失败", f"保存标定时出错：{str(e)}")

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
        id_item.setData(Qt.UserRole, mask_id)  # 存储mask_id
        id_item.setData(Qt.UserRole + 1, (r, g, b))  # 额外存储RGB值
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
        color_label.setFixedSize(40, 40)  # 增加主色块尺寸

        color_layout.addStretch()
        color_layout.addWidget(color_label)
        color_layout.addStretch()

        color_label.clicked.connect(lambda: self.show_color_dialog(row))
        self.annotation_table.setCellWidget(row, 1, color_container)

        # 占比列（不可编辑）
        percentage_item = QTableWidgetItem(f"{percentage:.1%}")
        percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
        percentage_item.setTextAlignment(Qt.AlignCenter)
        self.annotation_table.setItem(row, 2, percentage_item)

        # 可见性切换按钮
        visibility_btn = QPushButton()
        visibility_btn.setCheckable(True)
        visibility_btn.setChecked(True)
        visibility_btn.setFixedSize(50, 40)
        visibility_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: #28a745;
                font-family: "Segoe UI Symbol";
                font-size: 24px;
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

        visibility_widget = QWidget()
        visibility_layout = QHBoxLayout(visibility_widget)
        visibility_layout.addWidget(visibility_btn)
        visibility_layout.setAlignment(Qt.AlignCenter)
        visibility_layout.setContentsMargins(0, 0, 0, 0)
        self.annotation_table.setCellWidget(row, 3, visibility_widget)

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
        self.annotation_table.setCellWidget(row, 4, op_widget)

        # 设置行高
        self.annotation_table.setRowHeight(row, 50)  # 增加行高

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
        """更新颜色饼图，确保掩码占比总和为100%"""
        if self.viewer.cv_img is None:
            self.color_pie_chart_label.setText("无图片")
            return

        # 从masks字典中计算所有可见掩码的像素总数
        total_pixels = self.viewer.cv_img.shape[0] * self.viewer.cv_img.shape[1]  # 图像总像素
        colors_data = []
        mask_pixels = {}

        # 先计算每个掩码的像素数量和对应颜色
        for row in range(self.annotation_table.rowCount()):
            id_item = self.annotation_table.item(row, 0)
            if id_item:
                mask_id = id_item.data(Qt.UserRole)
                if mask_id and mask_id in self.viewer.masks:
                    mask_data = self.viewer.masks[mask_id]

                    # 只计算可见的掩码
                    if mask_data.get('visible', True):
                        mask = mask_data.get('mask')
                        mask_pixel_count = np.sum(mask)  # 掩码中的像素数

                        # 从ID项获取RGB值
                        r, g, b = id_item.data(Qt.UserRole + 1)

                        # 保存掩码数据
                        mask_pixels[mask_id] = {
                            'count': mask_pixel_count,
                            'rgb': (r, g, b)
                        }

        # 计算所有可见掩码的总像素数
        total_mask_pixels = sum(data['count'] for data in mask_pixels.values())

        if total_mask_pixels == 0:
            self.color_pie_chart_label.setText("无可见掩码")
            return

        # 计算每个掩码的占比（基于总掩码像素）
        for mask_id, data in mask_pixels.items():
            percentage = data['count'] / total_mask_pixels
            colors_data.append({
                'rgb': data['rgb'],
                'percentage': percentage
            })

        # 更新表格中的占比数据
        self.update_percentage_in_table(mask_pixels, total_mask_pixels)

        # 绘制饼图 - 优化字体设置
        # 尝试多种中文字体，适应不同操作系统
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                         'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC',
                         'Source Han Sans CN', 'Arial Unicode MS', 'sans-serif']

        # 使用matplotlib的font_manager检测可用字体
        import matplotlib.font_manager as fm
        import platform

        # 根据操作系统类型设置默认字体
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + plt.rcParams['font.sans-serif']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB'] + plt.rcParams['font.sans-serif']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC'] + plt.rcParams[
                'font.sans-serif']

        plt.rcParams['axes.unicode_minus'] = False

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))

        # 准备数据
        sizes = [data['percentage'] for data in colors_data]
        colors = [f"#{r:02x}{g:02x}{b:02x}" for data in colors_data for r, g, b in [data['rgb']]]

        # 绘制饼图
        wedges, texts = ax.pie(
            sizes,
            colors=colors,
            labels=[''] * len(colors),
            autopct=None,
            startangle=90
        )

        # 添加图例 - 使用英文避免中文显示问题
        legend_labels = [f'{data["percentage"]:.1%}' for data in colors_data]
        legend = ax.legend(
            wedges,
            legend_labels,
            title="Color Ratio",  # 使用英文"颜色占比"
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=14,
            title_fontsize=16
        )

        # 调整图例样式
        legend.get_frame().set_linewidth(2)
        legend.get_frame().set_edgecolor('black')

        ax.axis('equal')

        # 保存图片
        try:
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
        except Exception as e:
            print(f"[错误] 生成饼图时出错: {e}")
            import traceback
            print(traceback.format_exc())
            self.color_pie_chart_label.setText("饼图生成失败")

    def update_percentage_in_table(self, mask_pixels, total_mask_pixels):
        """更新表格中的占比数据"""
        for row in range(self.annotation_table.rowCount()):
            id_item = self.annotation_table.item(row, 0)
            if id_item:
                mask_id = id_item.data(Qt.UserRole)
                if mask_id in mask_pixels:
                    # 计算百分比
                    percentage = mask_pixels[mask_id]['count'] / total_mask_pixels

                    # 更新表格中的占比显示
                    percentage_item = self.annotation_table.item(row, 2)
                    if percentage_item:
                        percentage_item.setText(f"{percentage:.1%}")
                    else:
                        percentage_item = QTableWidgetItem(f"{percentage:.1%}")
                        percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
                        percentage_item.setTextAlignment(Qt.AlignCenter)
                        self.annotation_table.setItem(row, 2, percentage_item)

    def toggle_mask_visibility(self, row, checked):
        """切换掩码可见性"""
        try:
            # 从编号列获取 mask_id
            id_item = self.annotation_table.item(row, 0)
            if not id_item:
                return

            mask_id = id_item.data(Qt.UserRole)
            if not mask_id:
                return

            # 从ID项获取RGB值
            r, g, b = id_item.data(Qt.UserRole + 1)

            # 获取可见性按钮并更新显示
            visibility_widget = self.annotation_table.cellWidget(row, 3)
            if visibility_widget:
                visibility_btn = visibility_widget.findChild(QPushButton)
                if visibility_btn:
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
        # 获取当前文件所在目录
        current_dir = Path(__file__).resolve().parent
        default_dir = current_dir.parent / "images"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(default_dir),
            "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )
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
                            show_item = self.annotation_table.item(row, 3)
                            if show_item:
                                show_item.setText("✔" if visible else "✖")

                    # 更新预览
                    self.update_annotation_preview()
                    self.update_color_pie_chart()
                else:
                    # 清空显示
                    self.annotation_table.setRowCount(0)
                    self.annotation_preview_label.clear()
                    self.color_pie_chart_label.clear()
                    self.annotation_preview_label.setText("无标注数据")
                    self.color_pie_chart_label.setText("无标注数据")

            except Exception as e:
                print(f"[错误] 加载标注数据失败: {str(e)}")
                QMessageBox.warning(self, "加载失败", f"加载标注数据时出错：{str(e)}")
                # 出错时也清空显示
                self.annotation_table.setRowCount(0)
                self.annotation_preview_label.clear()
                self.color_pie_chart_label.clear()

            # 更新缩放UI
            self.update_scale_ui()
            slider_val = int(self.viewer.scale / self.viewer.base_scale * 100)
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(slider_val)
            self.scale_slider.blockSignals(False)

            # 在成功加载图像后添加
            if self.viewer.cv_img is not None:
                self.ai_segmentation.setImage(self.viewer.cv_img)

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
        show_item = self.annotation_table.item(row, 3)
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

        # 更新ID项中存储的RGB值
        id_item = self.annotation_table.item(row, 0)
        if id_item:
            id_item.setData(Qt.UserRole + 1, (r, g, b))

        # 更新占比
        percentage_item = QTableWidgetItem(f"{percentage:.1%}")
        percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
        percentage_item.setTextAlignment(Qt.AlignCenter)
        self.annotation_table.setItem(row, 2, percentage_item)

        # 更新颜色块 - 使用与新增时相同的 ClickableColorLabel
        color_container = QWidget()
        color_layout = QHBoxLayout(color_container)
        color_layout.setContentsMargins(0, 0, 0, 0)

        color_label = ClickableColorLabel()
        color_label.setStyleSheet(f"""
            background-color: rgb({r}, {g}, {b});
            border: 1px solid #dee2e6;
            border-radius: 2px;
        """)
        color_label.setFixedSize(40, 40)  # 与新增时保持一致的尺寸

        color_layout.addStretch()
        color_layout.addWidget(color_label)
        color_layout.addStretch()

        color_label.clicked.connect(lambda: self.show_color_dialog(row))
        self.annotation_table.setCellWidget(row, 1, color_container)

    def find_row_by_mask_id(self, mask_id):
        """根据 mask_id 查找对应的表格行"""
        for row in range(self.annotation_table.rowCount()):
            id_item = self.annotation_table.item(row, 0)  # 从第0列（编号列）获取
            if id_item and id_item.data(Qt.UserRole) == mask_id:
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
            # 从ID项获取存储的RGB值
            id_item = self.annotation_table.item(row, 0)
            if id_item and id_item.data(Qt.UserRole + 1):
                r, g, b = id_item.data(Qt.UserRole + 1)
            else:
                # 获取主色块的颜色
                color_container = self.annotation_table.cellWidget(row, 1)
                if color_container:
                    color_label = color_container.findChild(ClickableColorLabel)
                    if color_label:
                        style = color_label.styleSheet()
                        import re
                        rgb = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
                        if rgb:
                            r, g, b = map(int, rgb.groups())
                        else:
                            r, g, b = 0, 0, 0
                    else:
                        r, g, b = 0, 0, 0
                else:
                    r, g, b = 0, 0, 0

            # 创建颜色对话框
            color_dialog = QColorDialog(self)
            color_dialog.setCurrentColor(QColor(r, g, b))
            color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
            color_dialog.setWindowTitle("选择颜色")

            if color_dialog.exec_():
                # 获取新选择的颜色
                new_color = color_dialog.currentColor()
                new_r, new_g, new_b = new_color.red(), new_color.green(), new_color.blue()

                # 更新存储在ID项中的RGB值
                id_item.setData(Qt.UserRole + 1, (new_r, new_g, new_b))

                # 更新颜色块显示
                color_container = self.annotation_table.cellWidget(row, 1)
                if color_container:
                    color_label = color_container.findChild(ClickableColorLabel)
                    if color_label:
                        color_label.setStyleSheet(f"""
                            background-color: rgb({new_r}, {new_g}, {new_b});
                            border: 1px solid #dee2e6;
                            border-radius: 2px;
                        """)

                # 更新掩码颜色
                mask_id = id_item.data(Qt.UserRole)
                if mask_id in self.viewer.masks:
                    self.viewer.masks[mask_id]['color'] = (new_r, new_g, new_b)
                    self.viewer.update()

                # 更新颜色饼图
                self.update_color_pie_chart()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"修改颜色时出错：{str(e)}")

    def onAISegmentationCompleted(self, mask):
        """处理AI分割完成的结果"""
        try:
            print(f"[接收] 收到分割结果，掩码尺寸: {mask.shape}, 类型: {mask.dtype}")

            if self.viewer.cv_img is None:
                print("[警告] 没有加载图像，忽略分割结果")
                return

            if not isinstance(mask, np.ndarray):
                print(f"[错误] 掩码类型无效: {type(mask)}")
                return

            # 处理3D掩码（彩色掩码）
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                print("[处理] 检测到彩色掩码，分离不同颜色区域")

                # 增强彩色掩码清晰度
                img = self.viewer.cv_img.copy()

                # 创建颜色分类掩码字典
                color_masks = {}

                # 提取唯一颜色
                mask_flattened = mask.reshape(-1, 3)
                unique_colors = np.unique(mask_flattened, axis=0)

                # 移除黑色背景 [0,0,0]
                unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]

                print(f"[分析] 检测到 {len(unique_colors)} 种不同颜色")

                # 使用颜色相似度聚类算法对颜色进行合并
                if len(unique_colors) > 0:
                    # 转为浮点数进行K-means聚类
                    colors_float = unique_colors.astype(np.float32)

                    # 根据颜色数量确定聚类数
                    optimal_k = min(8, len(unique_colors))  # 最多保留8种颜色

                    if optimal_k > 1:  # 只有多于一种颜色时才进行聚类
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
                        _, labels, centers = cv2.kmeans(
                            colors_float, optimal_k, None, criteria,
                            10, cv2.KMEANS_PP_CENTERS
                        )

                        # 创建映射字典，将原始颜色映射到聚类中心
                        color_map = {}
                        for i, color in enumerate(unique_colors):
                            center_idx = labels[i][0]  # 修复：获取标签的第一个元素
                            center_color = tuple(map(int, centers[center_idx]))
                            color_map[tuple(color)] = center_color

                        # 重新着色掩码，使用聚类中心颜色
                        refined_mask = np.zeros_like(mask)
                        for y in range(mask.shape[0]):
                            for x in range(mask.shape[1]):
                                pixel = tuple(mask[y, x])
                                if pixel != (0, 0, 0):  # 跳过黑色背景
                                    if pixel in color_map:
                                        refined_mask[y, x] = color_map[pixel]

                        # 用合并后的颜色替换原始掩码
                        mask = refined_mask

                        # 重新提取唯一颜色
                        mask_flattened = mask.reshape(-1, 3)
                        unique_colors = np.unique(mask_flattened, axis=0)
                        unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]

                        print(f"[合并] 合并相似颜色后剩余 {len(unique_colors)} 种颜色")

                # 为每种颜色创建二值掩码
                for color in unique_colors:
                    # 创建该颜色的掩码
                    color_mask = np.all(mask == color.reshape(1, 1, 3), axis=2)

                    # 应用形态学操作清理掩码
                    kernel = np.ones((5, 5), np.uint8)
                    color_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

                    # 使用连通区域分析移除小区域
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
                    min_area = 500  # 最小连通区域面积阈值
                    refined_mask = np.zeros_like(color_mask)

                    for i in range(1, num_labels):  # 从1开始，跳过背景(0)
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            refined_mask[labels == i] = 1

                    # 检查掩码大小，忽略太小的区域
                    if np.sum(refined_mask) < 500:
                        continue

                    # 保存掩码
                    color_tuple = tuple(map(int, color))
                    color_masks[color_tuple] = refined_mask > 0

                # 后处理 - 确保掩码之间不重叠
                sorted_colors = sorted(color_masks.keys(),
                                       key=lambda c: np.sum(color_masks[c]),
                                       reverse=True)  # 按区域大小排序

                # 处理每个颜色掩码 - 使用原始图像颜色分析来优化颜色选择
                for color in sorted_colors:
                    color_mask = color_masks[color]

                    # 分析掩码区域的实际主要颜色
                    masked_img = img.copy()
                    masked_img[~color_mask] = 0  # 将非掩码区域设为黑色

                    # 使用颜色分析器分析该区域的主要颜色
                    color_infos = self.color_analyzer.analyze_image_colors(masked_img, color_mask, k=1)

                    if color_infos:
                        # 获取该区域的主要颜色
                        color_info = color_infos[0]

                        # 创建新的标注
                        mask_id = len(self.viewer.masks) if self.viewer.masks else 0

                        # 添加到标注列表
                        if self.viewer.masks is None:
                            self.viewer.masks = {}

                        print(f"[处理] 添加颜色 {color_info.rgb} 的掩码 ID: {mask_id}")
                        self.viewer.masks[mask_id] = {
                            'mask': color_mask,
                            'color': color_info.rgb,
                            'visible': True,
                            'editable': False
                        }

                        # 添加到表格
                        self.add_annotation_to_table(color_info, mask_id)

            # 处理二进制掩码（单一区域）
            elif len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[2] == 1):
                print("[分析] 处理二进制掩码")

                # 确保掩码是二维的
                if len(mask.shape) == 3:
                    mask = mask.squeeze()

                # 转换为布尔类型
                mask = mask.astype(bool)

                # 形态学操作清理掩码
                kernel = np.ones((5, 5), np.uint8)
                cleaned_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
                mask = cleaned_mask > 0

                # 使用连通区域分析，提取多个独立区域
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

                print(f"[分析] 检测到 {num_labels - 1} 个独立区域")

                # 分别处理每个区域
                for i in range(1, num_labels):  # 从1开始，跳过背景(0)
                    # 检查区域大小
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 500:  # 忽略小区域
                        continue

                    # 提取该区域的掩码
                    region_mask = (labels == i)

                    # 分析区域颜色
                    img = self.viewer.cv_img.copy()
                    masked_img = img.copy()
                    masked_img[~region_mask] = 0

                    color_infos = self.color_analyzer.analyze_image_colors(masked_img, region_mask)
                    if not color_infos:
                        continue

                    color_info = color_infos[0]

                    # 创建新的标注
                    mask_id = len(self.viewer.masks) if self.viewer.masks else 0

                    if self.viewer.masks is None:
                        self.viewer.masks = {}

                    print(f"[处理] 添加独立区域 {i}, 颜色: {color_info.rgb}, ID: {mask_id}")
                    self.viewer.masks[mask_id] = {
                        'mask': region_mask,
                        'color': color_info.rgb,
                        'visible': True,
                        'editable': False
                    }

                    # 添加到表格
                    self.add_annotation_to_table(color_info, mask_id)

            else:
                print(f"[错误] 不支持的掩码格式: shape={mask.shape}")
                return

            # 更新显示
            print("[更新] 刷新预览...")
            self.update_annotation_preview()
            self.update_color_pie_chart()
            print("[完成] 分割结果处理完成")

        except Exception as e:
            import traceback
            print(f"[错误] 处理分割结果时出错：\n{traceback.format_exc()}")
            QMessageBox.critical(self, "错误", f"处理分割结果时出错：{str(e)}")

    def mask_to_rle(self, mask):
        """将二进制掩码转换为RLE格式"""
        flat_mask = mask.flatten()
        starts = np.where(flat_mask[:-1] != flat_mask[1:])[0] + 1
        if flat_mask[0] == 1:
            starts = np.r_[0, starts]
        ends = np.where(flat_mask[:-1] != flat_mask[1:])[0] + 1
        if flat_mask[-1] == 1:
            ends = np.r_[ends, flat_mask.size]

        rle = []
        for start, end in zip(starts, ends):
            if flat_mask[start] == 1:
                rle.extend([int(start), int(end - start)])

        return rle

    def setupAISegmentation(self):
        """设置AI分割工具"""
        self.ai_segmentation = AISegmentationWidget(self)
        self.ai_segmentation.setViewer(self.viewer)  # 设置viewer引用

        # 共享SAM模型，避免重复加载
        if hasattr(self.viewer, 'sam') and self.viewer.sam:
            print("[优化] 共享SAM模型实例，避免重复加载")
            self.ai_segmentation.shared_sam_model = self.viewer.sam

        self.ai_segmentation.segmentation_completed.connect(self.onAISegmentationCompleted)

    def toggle_add_mode(self):
        """切换增加掩码模式"""
        if self.viewer.mode == "add":
            # 退出增加掩码模式
            self.viewer.set_add_mode()
            self.btn_add.setStyleSheet(self.button_style)  # 使用普通样式
        else:
            # 进入增加掩码模式
            self.viewer.set_add_mode()
            self.btn_add.setStyleSheet(self.active_button_style)  # 使用激活样式
            # 确保擦除按钮显示为普通样式
            self.btn_erase.setStyleSheet(self.button_style)

            # 强制更新视图
        self.viewer.update()

    def toggle_erase_mode(self):
        """切换擦除掩码模式"""
        if self.viewer.mode == "erase":
            # 退出擦除掩码模式
            self.viewer.set_erase_mode()
            self.btn_erase.setStyleSheet(self.button_style)  # 使用普通样式
        else:
            # 进入擦除掩码模式
            self.viewer.set_erase_mode()
            self.btn_erase.setStyleSheet(self.active_button_style)  # 使用激活样式
            # 确保增加按钮显示为普通样式
            self.btn_add.setStyleSheet(self.button_style)

        # 强制更新视图
        self.viewer.update()

    def check_magnifier_status(self):
        """定时检查放大镜状态并更新"""
        try:
            if not hasattr(self, 'viewer') or not self.viewer:
                return

            # 如果放大镜功能被禁用，直接返回
            if not self.magnifier_enabled:
                if self.magnifier_window.isVisible():
                    self.magnifier_window.hide()
                return

            # 优化：引入static变量实现检查节流
            if not hasattr(self, '_last_check_time'):
                self._last_check_time = 0

            current_time = int(time.time() * 1000)  # 转为毫秒
            if current_time - self._last_check_time < 80:  # 节流80ms
                return

            self._last_check_time = current_time

            # 检查放大镜是否应该显示
            if self.viewer.magnifier_active and self.viewer.mode in ("add", "erase"):
                # 获取当前鼠标位置并更新放大镜内容
                cursor_pos = self.viewer.mapFromGlobal(QCursor.pos())
                if self.viewer.rect().contains(cursor_pos):
                    # 手动调用更新放大镜内容的方法
                    self.viewer.update_magnifier(cursor_pos)
                    # 放大镜的显示/隐藏由信号处理
                else:
                    # 鼠标不在图像区域内，隐藏放大镜
                    self.magnifier_window.hide()
            else:
                # 放大镜不应该显示，隐藏窗口
                self.magnifier_window.hide()
        except Exception as e:
            print(f"[错误] 检查放大镜状态时出错: {e}")
            # 出错时尝试隐藏放大镜窗口
            try:
                self.magnifier_window.hide()
            except:
                pass

    def update_magnifier_preview(self, pixmap):
        """更新放大镜预览"""
        try:
            # 如果放大镜功能被禁用，直接返回
            if not self.magnifier_enabled:
                self.magnifier_window.clear()
                self.magnifier_window.hide()
                return

            if pixmap is None or pixmap.isNull():
                self.magnifier_window.clear()
                self.magnifier_window.hide()
                return

            # 保持适应窗口大小
            scaled_pixmap = pixmap.scaled(
                self.magnifier_window.width() - 10,  # 减小内边距
                self.magnifier_window.height() - 10,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.magnifier_window.setPixmap(scaled_pixmap)

            # 智能调整位置，避免遮挡当前操作区域
            cursor_pos = self.viewer.mapFromGlobal(QCursor.pos())
            window_w, window_h = self.magnifier_window.width(), self.magnifier_window.height()
            viewer_w, viewer_h = self.viewer.width(), self.viewer.height()

            # 默认位置在左上角
            pos_x, pos_y = 20, 20

            # 如果鼠标在左半部分，把放大镜放到右边
            if cursor_pos.x() < viewer_w / 2:
                pos_x = viewer_w - window_w - 20

            # 如果鼠标在上半部分，把放大镜放到下边
            if cursor_pos.y() < viewer_h / 2:
                pos_y = viewer_h - window_h - 20

            self.magnifier_window.move(pos_x, pos_y)
            self.magnifier_window.show()
        except Exception as e:
            print(f"[错误] 更新放大镜预览时出错: {e}")
            # 出错时尝试隐藏放大镜窗口
            try:
                self.magnifier_window.hide()
            except:
                pass

    def toggle_magnifier(self, state):
        """切换放大镜显示状态"""
        enabled = (state == Qt.Checked)
        self.magnifier_enabled = enabled  # 保存状态到类变量
        self.viewer.magnifier_active = enabled

        # 确保UI状态与选项状态一致
        if not enabled:
            # 强制隐藏放大镜窗口
            self.magnifier_window.hide()
            # 发送空的放大镜信号
            self.viewer.magnifierUpdated.emit(QPixmap())

        # 更新所有相关状态
        print(f"[设置] 放大镜显示状态: {'启用' if enabled else '禁用'}")

    def resizeEvent(self, event):
        """窗口大小变化时调整放大镜大小"""
        super().resizeEvent(event)
        self.update_magnifier_size()

    def update_magnifier_size(self):
        """根据窗口大小动态调整放大镜尺寸"""
        if hasattr(self, 'viewer') and self.viewer:
            # 根据查看器窗口宽度的一定比例设置放大镜大小
            viewer_width = self.viewer.width()
            viewer_height = self.viewer.height()

            # 计算合适的放大镜大小（查看器宽度的20%，但不小于240px且不大于400px）
            magnifier_size = int(min(max(viewer_width * 0.2, 240), 400))

            # 应用新尺寸
            if hasattr(self, 'magnifier_window'):
                self.magnifier_window.setFixedSize(magnifier_size, magnifier_size)
                print(f"[更新] 放大镜尺寸调整为: {magnifier_size}x{magnifier_size}像素")


# 添加可点击的颜色标签类
class ClickableColorLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针为手型

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()