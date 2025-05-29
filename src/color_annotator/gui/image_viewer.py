import json
import os

import numpy as np
import torch
import cv2
from PyQt5.QtWidgets import QLabel, QApplication, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QCursor, QColor
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal, QRect
from src.color_annotator.sam_interface.sam_segmentor import SAMSegmentor
from src.color_annotator.utils.sam_thread import SAMWorker  # 异步推理线程
from src.color_annotator.utils.color_analyzer import ColorAnalyzer  # 新增：颜色分析器
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import pyqtSignal


class ImageViewer(QLabel):
    scaleChanged = pyqtSignal(float)
    annotationAdded = pyqtSignal(tuple)  # 💡 新增发主色信号，(R, G, B)
    segmentationOverlayReady = pyqtSignal(QPixmap)  # 💡 发射 overlay 图

    def __init__(self):
        super().__init__()
        self.pending_mask_id = None
        self.add_button = None
        self.erase_button = None
        self.last_erase_pos = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray")
        self.mode = "normal"  # 默认为增加模式（可以修改为 'erase'）
        self.fg_points = []  # 前景点
        self.bg_points = []  # 背景点
        self.undo_stack = []
        self.redo_stack = []
        self.masks = {}  # { mask_id: {'mask': numpy.ndarray, 'visible': bool} }
        self.point_undo_stack = []  # 💡 添加：前景/背景点操作的撤销栈
        self.point_redo_stack = []
        self.eraser_shape_circle = False  # 默认使用方形橡皮擦

        self.scale_factor = None
        self.resized_image = None
        self.original_shape = None

        self.is_editing = False  # 当前是否正在一次连续编辑
        self.cv_img = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.dragging = False
        self.last_mouse_pos = QPoint(0, 0)
        self.base_scale = 1.0

        self.mask = None
        self.sam = SAMSegmentor(
            model_type="vit_b",  # 使用轻量模型
            checkpoint_path="checkpoints/sam_vit_b.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sam_thread = None  # 异步线程对象
        self.progress_dialog = None  # 加载中弹窗

        self.erase_size = 30  # 擦除区域的大小（正方形）

        self.erase_rect = None  # 用于存储擦除框的区域
        self.setFocusPolicy(Qt.StrongFocus)  # 💡 允许接受键盘焦点

        self.color_analyzer = ColorAnalyzer()  # 新增：颜色分析器实例

    def set_image(self, image: np.ndarray, max_size=512):
        self.cancel_segmentation()  # 如果有正在运行的分割线程，终止
        self.clear_annotations()  # 清除未保存的标定
        self.masks.clear()
        self.mask = None
        self.pending_mask_id = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.fg_points.clear()
        self.bg_points.clear()

        """设置图像供 SAM 使用，并自动 resize 控制大小"""
        self.original_shape = image.shape[:2]  # 原始大小 (h, w)
        h, w = self.original_shape

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            print(f"[自动缩放] 原图尺寸 {w}x{h} 已缩小为 {image.shape[1]}x{image.shape[0]}")

        self.cv_img = image  # 👈 直接将缩小图赋值为主图
        self.mask = None
        self.erase_size = int(min(image.shape[:2]) / 40)  # 动态调整橡皮擦大小（例如 25px）
        self.compute_initial_scale()
        self.reset_view()
        self.sam.set_image(image)  # SAM 使用缩小图

    def compute_initial_scale(self):
        if self.cv_img is None:
            return
        label_size = self.size()
        h, w = self.cv_img.shape[:2]
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        self.base_scale = min(scale_w, scale_h)
        self.scale = self.base_scale

    def set_add_mode(self):
        if self.mode == "add":
            # 当前是增加模式，点击后退出
            self.mode = "normal"
            print("[模式] 退出增加掩码模式，回到正常模式")
            self.setCursor(Qt.ArrowCursor)
            if self.add_button:
                self.add_button.setStyleSheet("")
        else:
            # 当前不是增加模式，点击进入增加模式
            self.mode = "add"
            print("[模式] 进入增加掩码模式")
            self.setCursor(Qt.BlankCursor)
            if self.add_button:
                self.add_button.setStyleSheet("background-color: lightgreen;")
            if self.erase_button:
                self.erase_button.setStyleSheet("")  # 取消擦除按钮高亮

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.cv_img is None:
            return

        painter = QPainter(self)
        rgb_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_img.shape
        bytes_per_line = channel * width

        # === 掩码处理开始 ===
        for mask_id, entry in self.masks.items():
            if not entry.get('visible', True):
                continue

            mask = entry['mask']
            color = entry.get('color', (0, 255, 0))
            r, g, b = color

            mask_bool = mask.astype(np.bool_)
            overlay = rgb_img.copy()
            # 如果当前是正在编辑的掩码，就使用高亮绿色显示
            if entry.get("editable", False) and mask_id == self.pending_mask_id:
                overlay[mask_bool] = (0, 255, 0)  # 高亮绿色
            else:
                overlay[mask_bool] = (r, g, b)

            # 半透明混合，仅对掩码区域有效
            alpha = 0.8
            rgb_img[mask_bool] = cv2.addWeighted(rgb_img[mask_bool], 1 - alpha, overlay[mask_bool], alpha, 0)

            # 添加边框轮廓
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 判断是否为已保存标定（editable=False）
            if entry.get('editable', False):
                border_color = (r, g, b)  # 当前编辑掩码使用主色边框
            else:
                border_color = (200, 200, 200)  # 已保存标定使用白色边框，提升可见性

            scale_thickness = max(1, int(1 / self.scale))  # 根据缩放动态调整线宽
            cv2.drawContours(rgb_img, contours, -1, border_color, thickness=scale_thickness)

        # === 转换为 QPixmap 并绘制图像 ===
        qt_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            int(self.cv_img.shape[1] * self.scale),
            int(self.cv_img.shape[0] * self.scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        draw_x = int((self.width() - pixmap.width()) / 2 + self.offset.x())
        draw_y = int((self.height() - pixmap.height()) / 2 + self.offset.y())
        painter.drawPixmap(int(draw_x), int(draw_y), pixmap)

        # === 点绘制：前景绿色、背景红色 ===
        scaled_w = int(self.cv_img.shape[1] * self.scale)
        scaled_h = int(self.cv_img.shape[0] * self.scale)
        draw_x = (self.width() - scaled_w) // 2 + self.offset.x()
        draw_y = (self.height() - scaled_h) // 2 + self.offset.y()

        painter.setBrush(Qt.green)
        for x, y in self.fg_points:
            px = int(x * self.scale + draw_x)
            py = int(y * self.scale + draw_y)
            painter.drawEllipse(QPoint(px, py), 5, 5)

        painter.setBrush(Qt.red)
        for x, y in self.bg_points:
            px = int(x * self.scale + draw_x)
            py = int(y * self.scale + draw_y)
            painter.drawEllipse(QPoint(px, py), 5, 5)

        # === 显示白色橡皮框 ===
        if self.mode in ("erase", "add"):
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            img_pos = self.map_to_image(cursor_pos)
            x, y = int(img_pos.x()), int(img_pos.y())
            half_size = self.erase_size // 2

            scaled_x = int(x * self.scale + draw_x)
            scaled_y = int(y * self.scale + draw_y)
            scaled_erase_size = int(self.erase_size * self.scale)

            # 画橡皮边框（灰色描边 + 白色填充）
            pen = painter.pen()
            pen.setWidth(1)
            pen.setColor(Qt.gray)  # 你可以改成 Qt.black 或 QColor(100, 100, 100)
            painter.setPen(pen)
            painter.setBrush(Qt.white)
            if self.eraser_shape_circle:
                painter.drawEllipse(QPoint(scaled_x, scaled_y), scaled_erase_size // 2, scaled_erase_size // 2)
            else:
                painter.drawRect(
                    scaled_x - scaled_erase_size // 2,
                    scaled_y - scaled_erase_size // 2,
                    scaled_erase_size,
                    scaled_erase_size
                )

    def mousePressEvent(self, event):
        self.setFocus()  # 鼠标点击时抢焦点，确保能按快捷键

        if self.mode in ("erase", "add"):
            if event.button() == Qt.LeftButton:
                # 开始新的编辑动作
                self.is_editing = True

                # 只在首次点击时记录 undo（若当前掩码是可编辑）
                if self.mask is not None and self.masks.get(self.pending_mask_id, {}).get("editable", False):
                    self.undo_stack.append(self.mask.copy())
                    self.redo_stack.clear()

                img_pos = self.map_to_image(event.pos())
                x, y = int(img_pos.x()), int(img_pos.y())
                print(f"[修改掩码] 当前模式: {self.mode}，位置: ({x}, {y})")
                self.modify_mask(x, y, save_history=True)  # 注意这里 save_history=False
        else:
            # 正常模式下添加前景点/背景点/拖动
            if event.button() == Qt.LeftButton:
                if event.modifiers() & Qt.ControlModifier:
                    # Ctrl + 左键 添加背景点
                    img_pos = self.map_to_image(event.pos())
                    x, y = int(img_pos.x()), int(img_pos.y())
                    print(f"[点击] 添加背景点：({x}, {y})")
                    self.bg_points.append((x, y))
                    self.point_undo_stack.append(("bg", (x, y)))  # 💡 添加撤销记录
                    self.point_redo_stack.clear()
                    self.repaint()
                else:
                    # 左键拖动
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)

            elif event.button() == Qt.RightButton:
                # 右键添加前景点
                img_pos = self.map_to_image(event.pos())
                x, y = int(img_pos.x()), int(img_pos.y())
                print(f"[点击] 添加前景点：({x}, {y})")
                self.fg_points.append((x, y))
                self.point_undo_stack.append(("fg", (x, y)))  # 💡 添加撤销记录
                self.point_redo_stack.clear()
                self.repaint()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                if not self.undo():  # 掩码无法撤销
                    self.undo_point()  # 尝试撤销前景点/背景点
                return
            elif event.key() == Qt.Key_Y:
                if not self.redo():
                    self.redo_point()
                return
        super().keyPressEvent(event)

    def undo(self):
        if self.undo_stack:
            if self.mask is not None:
                self.redo_stack.append(self.mask.copy())
            self.mask = self.undo_stack.pop()

            # 同步更新 masks 字典中的掩码数据
            if self.pending_mask_id and self.pending_mask_id in self.masks:
                self.masks[self.pending_mask_id]['mask'] = self.mask

            self.repaint()
            self.is_editing = False  # 撤销后强制结束当前编辑
            print("[撤销] 恢复到上一个掩码状态")
        else:
            print("[撤销] 无可撤销内容")

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.mask.copy())
            self.mask = self.redo_stack.pop()

            # 同步更新 masks 字典中的掩码数据
            if self.pending_mask_id and self.pending_mask_id in self.masks:
                self.masks[self.pending_mask_id]['mask'] = self.mask

            self.repaint()
            print("[重做] 恢复到撤销前的掩码状态")
        else:
            print("[重做] 无可恢复内容")

    def undo_point(self):
        if not self.point_undo_stack:
            print("[点撤销] 无可撤销内容")
            return False

        point_type, coord = self.point_undo_stack.pop()
        self.point_redo_stack.append((point_type, coord))
        if point_type == "fg":
            if coord in self.fg_points:
                self.fg_points.remove(coord)
        elif point_type == "bg":
            if coord in self.bg_points:
                self.bg_points.remove(coord)
        self.repaint()
        print(f"[点撤销] 撤销 {point_type} 点：{coord}")
        return True

    def redo_point(self):
        if not self.point_redo_stack:
            print("[点重做] 无可恢复内容")
            return False

        point_type, coord = self.point_redo_stack.pop()
        self.point_undo_stack.append((point_type, coord))
        if point_type == "fg":
            self.fg_points.append(coord)
        elif point_type == "bg":
            self.bg_points.append(coord)
        self.repaint()
        print(f"[点重做] 恢复 {point_type} 点：{coord}")
        return True

    def modify_mask(self, x, y, repaint=True, save_history=False):
        # 自动创建一个新的可编辑掩码（未经过分割）
        if self.mode == "add" and self.pending_mask_id is None:
            h, w = self.cv_img.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask_id = f"mask_{len(self.masks)}"

            self.masks[mask_id] = {
                'mask': mask,
                'visible': True,
                'color': (0, 255, 0),
                'editable': True
            }
            self.mask = mask
            self.pending_mask_id = mask_id

            color = self.extract_main_color()
            if color:
                self.annotationAdded.emit((color, mask_id))

        """根据当前模式，擦除或增加掩码"""
        if self.mask is None:
            h, w = self.cv_img.shape[:2]
            self.mask = np.zeros((h, w), dtype=bool)

        # 只处理 editable=True 的当前掩码（如果当前掩码不在 masks 中则默认允许）
        for mask_id, entry in self.masks.items():
            if entry.get("editable", False):
                break
        else:
            # 没有可编辑掩码，说明是只读状态，不进行修改
            return

        h, w = self.cv_img.shape[:2]
        half_size = self.erase_size // 2
        for i in range(max(0, y - half_size), min(h, y + half_size + 1)):
            for j in range(max(0, x - half_size), min(w, x + half_size + 1)):
                if 0 <= j < w and 0 <= i < h:
                    if self.eraser_shape_circle:
                        # 计算到中心点的距离
                        if (i - y) ** 2 + (j - x) ** 2 > half_size ** 2:
                            continue
                    if self.mode == "erase":
                        self.mask[i, j] = False
                    elif self.mode == "add":
                        self.mask[i, j] = True

        if repaint:
            self.repaint()

    def on_mask_ready(self, mask):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        if mask is None:
            print("[错误] 掩码生成失败")
            return

        if mask.shape[:2] != self.cv_img.shape[:2]:
            print(f"[错误] 掩码尺寸不合法：{mask.shape} vs 图像：{self.cv_img.shape}")
            return

        print(f"[完成] 掩码像素：{np.sum(mask)}")

        # 如果已有未保存的掩码，则合并
        if self.pending_mask_id and self.pending_mask_id in self.masks:
            print(f"[分割合并] 合并掩码到 {self.pending_mask_id}")
            self.masks[self.pending_mask_id]['mask'] |= mask
            self.mask = self.masks[self.pending_mask_id]['mask']
            # 清除标注点
            self.fg_points.clear()
            self.bg_points.clear()
            self.repaint()
            return

        # 否则新建一个掩码记录
        mask_id = f"mask_{len(self.masks)}"
        self.masks[mask_id] = {
            'mask': mask,
            'visible': True,
            'editable': True,
            'color': (0, 255, 0)  # 默认颜色，保存时会更新
        }
        self.mask = mask
        self.pending_mask_id = mask_id

        # 清除标注点
        self.fg_points.clear()
        self.bg_points.clear()
        self.repaint()

        # 提取颜色 & 发出 annotationAdded 信号
        color_info = self.extract_main_color()
        if color_info:
            self.annotationAdded.emit((color_info, mask_id))

        # 生成分割可视化图像
        self.generate_segmentation_overlay(mask)

    def generate_segmentation_overlay(self, mask):
        img = self.cv_img.copy()
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=2)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.segmentationOverlayReady.emit(pixmap)

    def extract_main_color(self):
        """从当前掩码提取主色（使用新的颜色分析器）"""
        if self.cv_img is None or self.mask is None:
            return None

        # 使用颜色分析器提取主色
        color_infos = self.color_analyzer.analyze_image_colors(
            self.cv_img, 
            self.mask,
            k=5  # 提取5个主要颜色
        )
        
        if not color_infos:
            return None
            
        # 返回占比最大的颜色信息对象
        return color_infos[0]  # 返回ColorInfo对象

    def set_mask_visibility(self, mask_id, visible, color=None):
        if mask_id in self.masks:
            self.masks[mask_id]['visible'] = visible

            if color is not None:
                self.masks[mask_id]['color'] = color  # 写入主色

            if visible:
                # 显示时必须设为不可编辑（只读模式）
                self.masks[mask_id]['editable'] = False
            else:
                # 隐藏时不强行设为只读
                self.masks[mask_id]['editable'] = self.masks[mask_id].get('editable', False)

            # 如果当前操作的是这个掩码，撤销它的编辑状态
            if self.pending_mask_id == mask_id:
                self.pending_mask_id = None
                self.mask = None

            self.update()

    def draw_line_between_points(self, x0, y0, x1, y1):
        """在两个点之间插值修改掩码"""
        dx = x1 - x0
        dy = y1 - y0
        distance = max(abs(dx), abs(dy))
        if distance == 0:
            self.modify_mask(x0, y0, repaint=False)
            return
        for i in range(distance + 1):
            x = int(x0 + dx * i / distance)
            y = int(y0 + dy * i / distance)
            self.modify_mask(x, y, repaint=False)

    def mouseMoveEvent(self, event):
        left_pressed = QApplication.mouseButtons() & Qt.LeftButton

        if self.mode in ("erase", "add"):
            if left_pressed and self.is_editing:
                img_pos = self.map_to_image(event.pos())
                x, y = int(img_pos.x()), int(img_pos.y())

                if self.last_erase_pos is not None:
                    last_x, last_y = self.last_erase_pos
                    self.draw_line_between_points(last_x, last_y, x, y)
                else:
                    self.modify_mask(x, y, save_history=True)  # 🚫

                self.last_erase_pos = (x, y)
            else:
                self.last_erase_pos = None

            self.update()
        else:
            if self.dragging and left_pressed:
                delta = event.pos() - self.last_mouse_pos
                self.offset += delta
                self.last_mouse_pos = event.pos()
                self.repaint()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_editing = False  # 结束编辑
            if self.dragging:
                self.dragging = False
                self.setCursor(Qt.ArrowCursor)

    def enterEvent(self, event):
        """鼠标进入控件"""
        if self.mode == "erase":
            self.setCursor(Qt.BlankCursor)

    def leaveEvent(self, event):
        """鼠标离开控件"""
        self.setCursor(Qt.ArrowCursor)  # 恢复正常箭头

    def set_add_button(self, button):
        self.add_button = button

    def set_erase_button(self, button):
        self.erase_button = button

    def set_erase_mode(self):
        if self.mode == "erase":
            # 当前是擦除模式，点击后退出
            self.mode = "normal"
            print("[模式] 退出擦除掩码模式，回到正常模式")
            self.setCursor(Qt.ArrowCursor)
            if self.erase_button:
                self.erase_button.setStyleSheet("")
        else:
            # 当前不是擦除模式，点击进入擦除模式
            self.mode = "erase"
            print("[模式] 进入擦除掩码模式")
            self.setCursor(Qt.BlankCursor)
            if self.erase_button:
                self.erase_button.setStyleSheet("background-color: lightblue;")
            if self.add_button:
                self.add_button.setStyleSheet("")  # 取消增加按钮高亮

    def update_mask(self, x, y):
        """更新掩码"""
        if self.mask is None:
            return

        h, w = self.cv_img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            if self.mode == "add":
                self.mask[y, x] = True  # 添加前景
            elif self.mode == "erase":
                self.mask[y, x] = False  # 擦除为背景

        self.repaint()  # 重新绘制掩码

    def zoom_image(self, event):
        old_pos = event.pos()
        img_pos_before = self.map_to_image(old_pos)
        self.scale *= 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        img_pos_after = self.map_to_image(old_pos)
        delta = img_pos_after - img_pos_before
        self.offset += QPoint(int(delta.x()), int(delta.y()))
        self.repaint()
        self.scaleChanged.emit(self.scale)

    def wheelEvent(self, event):
        ctrl_pressed = event.modifiers() & Qt.ControlModifier

        if ctrl_pressed:
            # Ctrl 被按下
            if self.mode in ("erase", "add"):
                # ✅ Ctrl + 滚轮 + 编辑模式 → 调整橡皮大小
                if event.angleDelta().y() > 0:
                    self.erase_size = min(self.erase_size + 2, 200)
                else:
                    self.erase_size = max(self.erase_size - 2, 5)
                print(f"[擦除大小调整] 当前橡皮擦大小: {self.erase_size}px")
                self.update()  # 刷新
            else:
                # Ctrl + 滚轮 + 非编辑模式 → 缩放
                self.zoom_image(event)
        else:
            # ✅ 普通滚轮：不论是否处于掩码模式，都可以缩放
            self.zoom_image(event)

    def set_scale(self, value):
        self.scale = value
        self.repaint()

    def get_scale_slider_value(self):
        return int(self.scale / self.base_scale * 100)

    def reset_view(self):
        self.scale = self.base_scale
        self.offset = QPoint(0, 0)
        self.repaint()

    def map_to_image(self, pos):
        if self.cv_img is None:
            return QPoint(0, 0)

        # 图像尺寸（缩放后）
        h, w = self.cv_img.shape[:2]
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)

        # 计算图像左上角位置（在控件内的位置）
        draw_x = (self.width() - scaled_w) // 2 + self.offset.x()
        draw_y = (self.height() - scaled_h) // 2 + self.offset.y()

        # 计算相对图像左上角的坐标（再除以缩放）
        relative_x = (pos.x() - draw_x) / self.scale
        relative_y = (pos.y() - draw_y) / self.scale

        return QPoint(int(relative_x), int(relative_y))

    # 取消分割
    def cancel_segmentation(self):
        if self.sam_thread and self.sam_thread.isRunning():
            print("[用户操作] 取消分割")
            self.sam_thread.terminate()
            self.sam_thread.wait()
        if self.progress_dialog:  # ✅ 加入空值判断
            self.progress_dialog.close()
            self.progress_dialog = None

    # 清空标定
    def clear_annotations(self):
        print("[清除] 清除所有未保存的标定区域")

        # 删除所有 editable=True 的掩码，保留已保存的
        new_masks = {}
        for mask_id, entry in self.masks.items():
            if not entry.get("editable", False):
                new_masks[mask_id] = entry

        self.masks = new_masks
        self.mask = None
        self.pending_mask_id = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.fg_points.clear()
        self.bg_points.clear()
        self.repaint()

    @staticmethod
    def decode_rle(rle: list, shape: tuple):
        flat = np.zeros(shape[0] * shape[1], dtype=bool)
        for start, length in rle:
            flat[start:start + length] = True
        return flat.reshape(shape)

    # 执行分割
    def run_sam_with_points(self):
        if not self.fg_points and not self.bg_points:
            print("[提示] 没有标注点")
            return

        # 过滤非法点
        h, w = self.cv_img.shape[:2]
        all_fg = [(x, y) for x, y in self.fg_points if 0 <= x < w and 0 <= y < h]
        all_bg = [(x, y) for x, y in self.bg_points if 0 <= x < w and 0 <= y < h]

        if not all_fg and not all_bg:
            print("[错误] 所有标注点无效")
            return

        input_points = np.array(all_fg + all_bg)
        input_labels = np.array([1] * len(all_fg) + [0] * len(all_bg))
        print(f"[执行] SAM 分割，点数={len(input_points)}，labels={input_labels.tolist()}")

        # 显示进度条
        self.progress_dialog = QProgressDialog("正在分割图像...", "取消", 0, 0, self)
        self.progress_dialog.setWindowTitle("请稍候")
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setCancelButtonText("取消")
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.canceled.connect(self.cancel_segmentation)
        self.progress_dialog.show()

        self.sam_thread = SAMWorker(
            self.sam,
            points=input_points,
            labels=input_labels
        )
        self.sam_thread.finished.connect(self.on_mask_ready)
        self.sam_thread.start()

    def load_masks_from_json(self, json_path):
        if not os.path.exists(json_path):
            print(f"[加载] 未找到标定文件：{json_path}")
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[错误] JSON 读取失败: {e}")
            return {}

        self.masks.clear()
        self.mask = None
        self.pending_mask_id = None
        self.undo_stack.clear()
        self.redo_stack.clear()

        # 新结构：annotations 列表
        annotations = data.get("annotations", [])
        loaded_masks = {}

        for idx, ann in enumerate(annotations):
            size = tuple(ann["size"])
            rle = ann["rle"]
            mask_array = self.decode_rle(rle, size)

            # ✅ 修复：若掩码尺寸与当前图像不一致，进行 resize（使用最近邻）
            if self.cv_img is not None and mask_array.shape != self.cv_img.shape[:2]:
                print(f"[修复] 掩码尺寸 {mask_array.shape} 与图像尺寸 {self.cv_img.shape[:2]} 不一致，自动缩放")
                mask_array = cv2.resize(
                    mask_array.astype(np.uint8),
                    (self.cv_img.shape[1], self.cv_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            color = tuple(ann.get("main_color", [0, 255, 0]))

            mask_id = f"mask_{idx}"
            self.masks[mask_id] = {
                "mask": mask_array,
                "color": color,
                "visible": True,
                "editable": False
            }
            loaded_masks[mask_id] = self.masks[mask_id]

        print(f"[加载] 成功载入 {len(self.masks)} 条掩码")
        self.repaint()
        return loaded_masks
