import os
import cv2
import numpy as np
import json
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QSlider, QLabel, QTableWidget, QAbstractItemView,
    QTableWidgetItem, QPushButton, QHeaderView, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("畲族服饰图像标定工具")
        self.setGeometry(100, 100, 1500, 800)

        self.viewer = ImageViewer()
        self.pending_annotation = None  # 分割后待保存的主色
        self.pending_mask_id = None     # 分割后掩码的 ID

        # === 操作栏按钮 ===
        open_btn = QPushButton("打开图片")
        open_btn.clicked.connect(self.open_image)

        reset_btn = QPushButton("重置视图")
        reset_btn.clicked.connect(self.viewer.reset_view)

        run_btn = QPushButton("执行分割")
        run_btn.clicked.connect(self.run_segmentation)

        save_btn = QPushButton("保存标定")
        save_btn.clicked.connect(self.save_annotation)

        clear_btn = QPushButton("清除标注")
        clear_btn.clicked.connect(self.viewer.clear_annotations)

        add_btn = QPushButton("增加掩码")
        add_btn.clicked.connect(self.viewer.set_add_mode)

        erase_btn = QPushButton("擦除掩码")
        erase_btn.clicked.connect(self.viewer.set_erase_mode)

        self.viewer.set_erase_button(erase_btn)
        self.viewer.set_add_button(add_btn)

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

        save_file_btn = QPushButton("保存数据")
        save_file_btn.clicked.connect(self.save_annotations_to_json)
        control_bar.addWidget(save_file_btn)

        control_bar.setContentsMargins(10, 10, 10, 0)
        control_bar.setSpacing(15)
        control_bar.addWidget(open_btn)
        control_bar.addWidget(reset_btn)
        control_bar.addWidget(zoom_label)
        control_bar.addWidget(self.scale_slider)
        control_bar.addWidget(self.scale_label)
        control_bar.addStretch()
        control_bar.addWidget(run_btn)
        control_bar.addWidget(save_btn)
        control_bar.addWidget(clear_btn)
        control_bar.addWidget(add_btn)
        control_bar.addWidget(erase_btn)

        control_widget = QWidget()
        control_widget.setLayout(control_bar)

        # === 右侧标注表格 ===
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(7)
        self.annotation_table.setHorizontalHeaderLabels(["编号", "主色", "R", "G", "B", "显示", "操作"])
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)

        # 设定每列固定宽度（单位：像素）
        self.annotation_table.setColumnWidth(0, 50)  # 编号
        self.annotation_table.setColumnWidth(1, 60)  # 主色
        self.annotation_table.setColumnWidth(2, 50)  # R
        self.annotation_table.setColumnWidth(3, 50)  # G
        self.annotation_table.setColumnWidth(4, 50)  # B
        self.annotation_table.setColumnWidth(5, 80)  # 显示
        self.annotation_table.setColumnWidth(6, 120)  # 操作

        # 总宽度 = 所有列宽 + 预留间距，防止横向滚动条出现
        total_width = 50 + 60 + 50 + 50 + 50 + 80 + 120 + 20
        self.annotation_table.setMinimumWidth(total_width)
        self.annotation_table.setMaximumWidth(total_width)

        # 设置样式：禁用选中变色/点击变透明
        self.annotation_table.setStyleSheet("""
        QTableWidget::item:selected {
            background-color: transparent;
            color: black;
        }
        QTableWidget::item:focus {
            background-color: transparent;
            color: black;
        }
        QTableWidget::item:hover {
            background-color: transparent;
        }
        """)

        # 禁止自动拉伸防止破坏布局
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)

        # 单元格点击事件绑定（如显示切换）
        self.annotation_table.cellClicked.connect(self.toggle_mask_visibility)
        self.annotation_table.itemChanged.connect(self.handle_color_change)

        # === 左侧主布局 ===
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(control_widget)
        left_layout.addWidget(self.viewer)
        left_layout.setStretch(0, 0)
        left_layout.setStretch(1, 1)

        # === 整体布局 ===
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.annotation_table)
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.viewer.scaleChanged.connect(self.update_scale_ui)
        self.viewer.annotationAdded.connect(self.cache_annotation)

    def run_segmentation(self):
        # 清除旧的 pending 掩码状态
        self.pending_annotation = None
        self.pending_mask_id = None
        self.viewer.pending_mask_id = None
        self.viewer.mask = None  # 防止继续合并老掩码

        self.viewer.run_sam_with_points()

    def cache_annotation(self, color_and_mask):
        color, mask_id = color_and_mask
        self.pending_annotation = color
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
            rgb = cv2.cvtColor(self.viewer.cv_img, cv2.COLOR_BGR2RGB)
            pixels = rgb[mask_data]
            if pixels.size > 0:
                color = tuple(pixels.mean(axis=0).astype(int).tolist())
                self.pending_annotation = color
            else:
                print("[错误] 无法提取颜色（掩码区域无像素）")
                return

        # 显示前先写入颜色
        self.viewer.masks[self.pending_mask_id]['color'] = self.pending_annotation
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

    def add_annotation_to_table(self, color, mask_id):
        row = self.annotation_table.rowCount()
        self.annotation_table.insertRow(row)

        r, g, b = color

        # 编号列（不可编辑）
        id_item = QTableWidgetItem(str(row + 1))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        self.annotation_table.setItem(row, 0, id_item)

        # 主色块 QLabel（不可编辑）
        color_label = QLabel()
        color_label.setStyleSheet(f"""
            background-color: rgb({r}, {g}, {b});
            border: 1px solid gray;
            margin-left: 5px;
            margin-right: 5px;
        """)
        color_label.setFixedSize(40, 20)
        color_label.setAlignment(Qt.AlignCenter)
        self.annotation_table.setCellWidget(row, 1, color_label)

        # R/G/B 列（可编辑）
        for col, val in zip((2, 3, 4), (r, g, b)):
            item = QTableWidgetItem(str(val))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.annotation_table.setItem(row, col, item)

        # 获取实际是否显示（默认为 True）
        visible = self.viewer.masks.get(mask_id, {}).get("visible", True)
        show_item = QTableWidgetItem("✔" if visible else "✖")

        show_item.setTextAlignment(Qt.AlignCenter)
        show_item.setFlags(show_item.flags() & ~Qt.ItemIsEditable)
        show_item.setData(Qt.UserRole, mask_id)
        self.annotation_table.setItem(row, 5, show_item)

        # 操作列：修改+删除按钮（不可编辑）
        edit_btn = QPushButton("修改")
        del_btn = QPushButton("删除")

        def handle_edit():
            self.edit_annotation(row, mask_id)

        edit_btn.clicked.connect(handle_edit)

        def handle_delete():
            self.delete_annotation(mask_id, row)

        del_btn.clicked.connect(handle_delete)

        op_widget = QWidget()
        op_layout = QHBoxLayout()
        op_layout.addWidget(edit_btn)
        op_layout.addWidget(del_btn)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.setSpacing(5)
        op_widget.setLayout(op_layout)
        self.annotation_table.setCellWidget(row, 6, op_widget)

        # 设置每行高度
        self.annotation_table.setRowHeight(row, 30)

    def delete_annotation(self, mask_id, row):
        reply = QMessageBox.question(
            self,
            "确认删除",
            "是否确定删除该标定？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if mask_id in self.viewer.masks:
                del self.viewer.masks[mask_id]
            self.annotation_table.removeRow(row)
            self.viewer.repaint()
            print(f"[删除] 掩码 {mask_id} 已删除")

            for i in range(self.annotation_table.rowCount()):
                id_item = self.annotation_table.item(i, 0)
                if id_item:
                    id_item.setText(str(i + 1))
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

    def toggle_mask_visibility(self, row, column):
        if column == 5:
            item = self.annotation_table.item(row, column)
            mask_id = item.data(Qt.UserRole)  # 正确读取绑定的 mask_id

            if not mask_id:
                return

            if item.text() == "✔":
                item.setText("✖")
                self.viewer.set_mask_visibility(mask_id, False)
            else:
                item.setText("✔")

                # 从当前行提取 R/G/B 值
                r = int(self.annotation_table.item(row, 2).text())
                g = int(self.annotation_table.item(row, 3).text())
                b = int(self.annotation_table.item(row, 4).text())

                # 显示掩码时传入主色
                self.viewer.set_mask_visibility(mask_id, True, (r, g, b))

            self.annotation_table.clearSelection()  # 取消高亮蓝色

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
        if self.viewer.cv_img is None:
            QMessageBox.warning(self, "未加载图像", "请先打开一张图片。")
            return

        # 默认保存到 annotations/ 同名 .json
        if not hasattr(self.viewer, 'image_path') or not self.viewer.image_path:
            QMessageBox.warning(self, "保存失败", "无法确定图像路径。")
            return

        base_name = os.path.splitext(os.path.basename(self.viewer.image_path))[0]
        save_path = os.path.join("annotations", base_name + ".json")
        os.makedirs("annotations", exist_ok=True)

        abs_img_path = self.viewer.image_path
        proj_root = os.path.abspath(os.getcwd())
        rel_img_path = os.path.relpath(abs_img_path, proj_root)

        data = {
            "image_path": rel_img_path.replace("\\", "/"),  # 替换为 Linux 风格
            "annotations": []
        }

        for mask_id, entry in self.viewer.masks.items():
            mask_array = entry["mask"].astype(int)
            rle = self.encode_rle(mask_array)
            height, width = mask_array.shape

            data["annotations"].append({
                "rle": rle,
                "size": [height, width],
                "main_color": list(entry.get("color", [0, 255, 0]))
            })

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "保存成功", f"标定数据已保存至：\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"发生错误：{str(e)}")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            self.viewer.set_image(img)

            self.viewer.image_path = file_path

            # === 自动加载对应 JSON 文件 ===
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join("annotations", base_name + ".json")
            masks = self.viewer.load_masks_from_json(json_path)

            # === 同步表格显示 ===
            self.annotation_table.setRowCount(0)
            for mask_id, entry in masks.items():
                color = entry["color"]
                self.add_annotation_to_table(color, mask_id)

                # 设置掩码显示状态（默认显示）
                visible = entry.get("visible", True)
                self.viewer.set_mask_visibility(mask_id, visible)

                # 找到对应行，设置显示列为 ✔ / ✖
                row = self.find_row_by_mask_id(mask_id)
                if row != -1:
                    show_item = self.annotation_table.item(row, 5)
                    if show_item:
                        show_item.setText("✔" if visible else "✖")

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
        show_item = self.annotation_table.item(row, 5)
        if show_item:
            mask_id = show_item.data(Qt.UserRole)
            if mask_id and mask_id in self.viewer.masks:
                self.viewer.masks[mask_id]['color'] = (r, g, b)
                self.viewer.update()

    def slider_zoom(self, value):
        scale = self.viewer.base_scale * (value / 100.0)
        self.viewer.set_scale(scale)
        self.update_scale_ui()

    def update_annotation_row(self, row, color, mask_id):
        r, g, b = color
        self.annotation_table.setItem(row, 2, QTableWidgetItem(str(r)))
        self.annotation_table.setItem(row, 3, QTableWidgetItem(str(g)))
        self.annotation_table.setItem(row, 4, QTableWidgetItem(str(b)))

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
            show_item = self.annotation_table.item(row, 5)
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
