import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import os
import time

import numpy as np
import torch
import cv2
from PyQt5.QtWidgets import QLabel, QApplication, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QCursor, QColor
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal, QRect, QTimer
from src.color_annotator.sam_interface.sam_segmentor import SAMSegmentor
from src.color_annotator.utils.sam_thread import SAMWorker  # 异步推理线程
from src.color_annotator.utils.color_analyzer import ColorAnalyzer  # 新增：颜色分析器
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import pyqtSignal


class ImageViewer(QLabel):
    scaleChanged = pyqtSignal(float)
    annotationAdded = pyqtSignal(tuple)  # 💡 新增发主色信号，(R, G, B)
    magnifierUpdated = pyqtSignal(QPixmap)  # 新增放大镜信号

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化SAM分割器
        try:
            self.sam = SAMSegmentor()
            print("[Viewer] SAM分割器初始化成功")
        except Exception as e:
            print(f"[Viewer] SAM分割器初始化失败: {str(e)}")
            self.sam = None
        
        # 基本属性
        self.cv_img = None
        self.pixmap = None
        self.scale = 1.0
        self.base_scale = 1.0
        self.offset = QPoint(0, 0)
        self.last_pos = None
        
        # 绘制相关
        self.drawing = False
        self.erasing = False
        self.mask = None
        self.masks = {}
        self.pending_mask_id = None
        self.eraser_size = 20
        self.eraser_shape_circle = True
        
        # 点击和标注
        self.click_points = []
        self.click_labels = []
        self.fg_points = []
        self.bg_points = []
        
        # 按钮和模式
        self.add_button = None
        self.erase_button = None
        self.last_erase_pos = None
        self.mode = "normal"
        
        # 撤销/重做
        self.undo_stack = []
        self.redo_stack = []
        self.point_undo_stack = []
        self.point_redo_stack = []
        
        # 缩放和变换
        self.scale_factor = None
        self.resized_image = None
        self.original_shape = None
        self.dragging = False
        self.last_mouse_pos = QPoint(0, 0)
        
        # 编辑状态
        self.is_editing = False
        self.erase_rect = None
        
        # 线程和进度
        self.sam_thread = None
        self.progress_dialog = None
        
        # 颜色分析器
        self.color_analyzer = ColorAnalyzer()
        
        # 放大镜相关 - 调整参数
        self.magnifier_active = False
        self.magnifier_zoom = 2.5  # 减小放大倍数以容纳更大区域
        self.magnifier_size = 180  # 增加放大镜尺寸
        self.current_magnifier_pixmap = None  # 存储当前放大镜内容
        self.magnifier_update_delay = 20  # 降低放大镜更新延迟(ms)，提高响应
        self.last_magnifier_update = time.time()
        self.pending_update = False  # 是否有待处理的更新
        
        # 性能优化 - 添加绘制节流变量
        self.last_paint_time = 0
        self.paint_throttle_ms = 15  # 降低绘制间隔时间(毫秒)，确保更流畅的体验
        
        # 界面设置
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray")

    def set_image(self, image: np.ndarray, max_size=1024):
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

        # 只有在图像特别大时才进行缩放
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            # 使用更好的插值方法
            image = cv2.resize(image, (int(w * scale), int(h * scale)), 
                             interpolation=cv2.INTER_LANCZOS4)
            print(f"[自动缩放] 原图尺寸 {w}x{h} 已缩小为 {image.shape[1]}x{image.shape[0]}")

        self.cv_img = image
        self.mask = None
        self.eraser_size = int(min(image.shape[:2]) / 40)  # 动态调整橡皮擦大小
        self.compute_initial_scale()
        self.reset_view()
        self.sam.set_image(image)  # SAM 使用处理后的图像

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
            # 只在放大镜处于活动状态时才关闭它
            if self.magnifier_active:
                # 触发信号通知主窗口隐藏放大镜
                self.magnifierUpdated.emit(QPixmap())
        else:
            # 当前不是增加模式，点击进入增加模式
            self.mode = "add"
            print("[模式] 进入增加掩码模式")
            self.setCursor(Qt.BlankCursor)
            if self.add_button:
                self.add_button.setStyleSheet("background-color: lightgreen;")
            if self.erase_button:
                self.erase_button.setStyleSheet("")  # 取消擦除按钮高亮
            # 只在放大镜处于活动状态时才触发它
            if self.magnifier_active:
                # 不改变magnifier_active状态，只根据当前状态通知UI
                # 在mouseMoveEvent中会更新放大镜内容
                pass

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
            half_size = self.eraser_size // 2

            scaled_x = int(x * self.scale + draw_x)
            scaled_y = int(y * self.scale + draw_y)
            scaled_erase_size = int(self.eraser_size * self.scale)

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
                self.magnifier_active = True  # 激活放大镜

                # 只在首次点击时记录 undo（若当前掩码是可编辑）
                if self.mask is not None and self.masks.get(self.pending_mask_id, {}).get("editable", False):
                    self.undo_stack.append(self.mask.copy())
                    self.redo_stack.clear()

                img_pos = self.map_to_image(event.pos())
                x, y = int(img_pos.x()), int(img_pos.y())
                print(f"[修改掩码] 当前模式: {self.mode}，位置: ({x}, {y})")
                self.modify_mask(x, y, save_history=True)
                
                # 更新放大镜
                self.update_magnifier(event.pos())
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
        try:
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

            # 检查坐标是否在图像范围内
            h, w = self.cv_img.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return

            # 只处理 editable=True 的当前掩码（如果当前掩码不在 masks 中则默认允许）
            for mask_id, entry in self.masks.items():
                if entry.get("editable", False):
                    break
            else:
                # 没有可编辑掩码，说明是只读状态，不进行修改
                return

            half_size = self.eraser_size // 2
            
            # 创建一个临时掩码，用于圆形橡皮擦
            if self.eraser_shape_circle:
                try:
                    # 计算圆形区域的边界，确保在图像范围内
                    x1 = max(0, x - half_size)
                    y1 = max(0, y - half_size)
                    x2 = min(w, x + half_size)
                    y2 = min(h, y + half_size)
                    
                    if x2 <= x1 or y2 <= y1:  # 无效区域
                        return
                        
                    # 提取当前区域的子掩码以提高性能
                    sub_h, sub_w = y2-y1, x2-x1
                    sub_center_x = x - x1
                    sub_center_y = y - y1
                    
                    # 创建圆形掩码
                    y_indices, x_indices = np.ogrid[:sub_h, :sub_w]
                    dist_from_center = np.sqrt((x_indices - sub_center_x)**2 + (y_indices - sub_center_y)**2)
                    circle_mask = dist_from_center <= half_size
                    
                    # 应用掩码
                    if self.mode == "erase":
                        self.mask[y1:y2, x1:x2][circle_mask] = False
                    elif self.mode == "add":
                        self.mask[y1:y2, x1:x2][circle_mask] = True
                except Exception as e:
                    print(f"[错误] 圆形橡皮擦失败: {e}")
                    self.eraser_shape_circle = False
            
            # 方形橡皮擦或圆形失败时的备选方案
            if not self.eraser_shape_circle:
                # 计算方形区域的边界，确保在图像范围内
                x1 = max(0, x - half_size)
                y1 = max(0, y - half_size)
                x2 = min(w, x + half_size)
                y2 = min(h, y + half_size)
                
                # 方形橡皮擦：使用矩阵操作代替遍历
                if self.mode == "erase":
                    self.mask[y1:y2, x1:x2] = False
                elif self.mode == "add":
                    self.mask[y1:y2, x1:x2] = True

            # 同步更新masks字典
            if self.pending_mask_id in self.masks:
                self.masks[self.pending_mask_id]['mask'] = self.mask
                
            # 避免频繁重绘，减少性能消耗，但降低阈值以提高流畅度
            current_time = time.time() * 1000  # 转换为毫秒
            if repaint and current_time - self.last_paint_time > self.paint_throttle_ms:
                self.last_paint_time = current_time
                self.update()
                
        except Exception as e:
            print(f"[错误] 修改掩码时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def on_mask_ready(self, mask):
        """处理分割完成的回调"""
        try:
            print("[分割] 收到分割结果")
            
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

            if mask is None:
                print("[错误] 掩码生成失败")
                return

            if not isinstance(mask, np.ndarray):
                print(f"[错误] 掩码类型不正确: {type(mask)}")
                return

            print(f"[分割] 掩码尺寸: {mask.shape}, 类型: {mask.dtype}")
            print(f"[分割] 掩码统计: 最小值={mask.min()}, 最大值={mask.max()}, 平均值={mask.mean():.4f}")
            print(f"[分割] 掩码中前景像素数: {np.sum(mask)}")

            if mask.shape[:2] != self.cv_img.shape[:2]:
                print(f"[错误] 掩码尺寸不合法：{mask.shape} vs 图像：{self.cv_img.shape}")
                return

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
            print(f"[分割] 创建新掩码: {mask_id}")
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
            print("[分割] 提取主色...")
            color_info = self.extract_main_color()
            if color_info:
                print(f"[分割] 主色: RGB={color_info.rgb}, 占比={color_info.percentage:.1%}")
                self.annotationAdded.emit((color_info, mask_id))
            else:
                print("[警告] 无法提取主色")

            print("[分割] 处理完成")

        except Exception as e:
            import traceback
            print(f"[错误] 处理分割结果时出错: {str(e)}")
            print(traceback.format_exc())

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
        """在两个点之间插值修改掩码，使用NumPy向量化操作提高性能"""
        try:
            dx = x1 - x0
            dy = y1 - y0
            distance = max(abs(dx), abs(dy))
            
            if distance == 0:
                self.modify_mask(x0, y0, repaint=False)
                return
            
            # 增加插值点数量，确保线条连续性
            # 对于所有距离都使用足够多的点来保证连续性
            steps = min(distance * 2, 50)  # 增加插值点数量，但设置上限避免过多计算
            
            # 使用NumPy生成插值点序列
            t = np.linspace(0, 1, int(steps))
            x_points = np.array(x0 + dx * t, dtype=int)
            y_points = np.array(y0 + dy * t, dtype=int)
            
            # 遍历插值点，每隔几个点重绘一次以保持性能
            for i, (x, y) in enumerate(zip(x_points, y_points)):
                self.modify_mask(x, y, repaint=(i % 5 == 0))  # 每5个点更新一次显示
            
            # 确保最后一个点被绘制并刷新显示
            self.modify_mask(x1, y1, repaint=True)
                
        except Exception as e:
            print(f"[错误] 绘制线条时出错: {str(e)}")

    def mouseMoveEvent(self, event):
        try:
            left_pressed = QApplication.mouseButtons() & Qt.LeftButton

            if self.mode in ("erase", "add"):
                # 无论是否在编辑状态，都需要更新橡皮擦位置显示
                self.setCursor(Qt.BlankCursor)  # 确保光标隐藏
                
                # 只有在放大镜处于活动状态时才更新放大镜
                if self.magnifier_active:
                    # 更新放大镜，使用节流控制频率
                    current_time = time.time()
                    if current_time - self.last_magnifier_update > self.magnifier_update_delay / 1000:
                        try:
                            self.update_magnifier(event.pos())
                        except Exception as e:
                            print(f"[警告] 更新放大镜时出错: {e}")
                    else:
                        self.pending_update = True
                
                # 强制更新绘制以显示橡皮擦位置
                self.update()
                
                # 处理绘制操作
                if left_pressed and self.is_editing:
                    img_pos = self.map_to_image(event.pos())
                    x, y = int(img_pos.x()), int(img_pos.y())

                    # 检查坐标是否有效
                    h, w = self.cv_img.shape[:2] if self.cv_img is not None else (0, 0)
                    if not (0 <= x < w and 0 <= y < h):
                        return

                    if self.last_erase_pos is not None:
                        last_x, last_y = self.last_erase_pos
                        # 减少阈值，确保更小的移动也能被捕获
                        self.draw_line_between_points(last_x, last_y, x, y)
                        self.last_erase_pos = (x, y)
                    else:
                        self.modify_mask(x, y, save_history=False)
                        self.last_erase_pos = (x, y)

            else:
                if self.dragging and left_pressed:
                    delta = event.pos() - self.last_mouse_pos
                    self.offset += delta
                    self.last_mouse_pos = event.pos()
                    self.repaint()

            super().mouseMoveEvent(event)
        except Exception as e:
            print(f"[错误] 鼠标移动事件处理出错: {e}")
            import traceback
            print(traceback.format_exc())

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.is_editing = False  # 结束编辑
                
                if self.mode not in ("add", "erase"):
                    # 只有在非编辑模式下才关闭放大镜
                    if self.magnifier_active:
                        try:
                            self.magnifierUpdated.emit(QPixmap())
                        except Exception as e:
                            print(f"[警告] 发送放大镜信号时出错: {e}")
                
                self.last_erase_pos = None  # 重置擦除位置
                
                if self.dragging:
                    self.dragging = False
                    self.setCursor(Qt.ArrowCursor)
                
                # 强制重绘
                self.update()
        except Exception as e:
            print(f"[错误] 鼠标释放事件处理出错: {e}")
            import traceback
            print(traceback.format_exc())

    def enterEvent(self, event):
        """鼠标进入控件"""
        if self.mode in ("add", "erase"):
            self.setCursor(Qt.BlankCursor)
            # 不应该在此处改变magnifier_active状态，而是使用已有状态

    def leaveEvent(self, event):
        """鼠标离开控件"""
        self.setCursor(Qt.ArrowCursor)  # 恢复正常箭头
        if self.mode in ("add", "erase") and self.magnifier_active:
            # 只在已启用放大镜的情况下发送隐藏信号
            try:
                self.magnifierUpdated.emit(QPixmap())
            except Exception as e:
                print(f"[警告] 发送放大镜信号时出错: {e}")

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
            # 只在放大镜处于活动状态时才关闭它
            if self.magnifier_active:
                # 触发信号通知主窗口隐藏放大镜
                self.magnifierUpdated.emit(QPixmap())
        else:
            # 当前不是擦除模式，点击进入擦除模式
            self.mode = "erase"
            print("[模式] 进入擦除掩码模式")
            self.setCursor(Qt.BlankCursor)
            if self.erase_button:
                self.erase_button.setStyleSheet("background-color: lightblue;")
            if self.add_button:
                self.add_button.setStyleSheet("")  # 取消增加按钮高亮
            # 只在放大镜处于活动状态时才触发它
            if self.magnifier_active:
                # 不改变magnifier_active状态，只根据当前状态通知UI
                # 在mouseMoveEvent中会更新放大镜内容
                pass

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
                    self.eraser_size = min(self.eraser_size + 2, 200)
                else:
                    self.eraser_size = max(self.eraser_size - 2, 5)
                print(f"[擦除大小调整] 当前橡皮擦大小: {self.eraser_size}px")
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
        """执行基于前景点和背景点的SAM分割"""
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
        print(f"[执行] SAM 分割，前景点={len(all_fg)}，背景点={len(all_bg)}")

        # 显示进度条
        self.progress_dialog = QProgressDialog("正在分割图像...", "取消", 0, 0, self)
        self.progress_dialog.setWindowTitle("请稍候")
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setCancelButtonText("取消")
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.canceled.connect(self.cancel_segmentation)
        self.progress_dialog.show()

        # 使用原始SAM进行分割
        self.sam_thread = SAMWorker(
            self.sam,
            points=input_points,
            labels=input_labels,
            multimask_output=False  # 只输出一个掩码
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

    def edit_annotation(self, mask_id):
        """进入编辑模式"""
        print(f"[编辑] 开始编辑掩码 {mask_id}")
        if mask_id in self.masks:
            # 保存当前颜色
            current_color = self.masks[mask_id].get('color', (0, 255, 0))
            
            # 设置该掩码为可编辑状态
            self.masks[mask_id]['editable'] = True
            self.masks[mask_id]['visible'] = True
            self.masks[mask_id]['color'] = current_color  # 确保保持原有颜色
            
            # 设置为当前活动掩码
            self.pending_mask_id = mask_id
            self.mask = self.masks[mask_id]['mask'].copy()  # 创建副本以防止直接修改
            
            # 清空撤销/重做栈
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            print(f"[编辑] 掩码设置完成，颜色: {current_color}")
            self.update()

    def get_magnifier_pixmap(self):
        """获取当前放大镜内容"""
        return self.current_magnifier_pixmap

    def update_magnifier(self, pos):
        """更新放大镜预览"""
        try:
            current_time = time.time()
            # 节流控制，避免频繁更新放大镜
            if current_time - self.last_magnifier_update < self.magnifier_update_delay / 1000:
                self.pending_update = True
                return
                
            self.last_magnifier_update = current_time
            self.pending_update = False
            
            if not self.magnifier_active or self.cv_img is None:
                self.current_magnifier_pixmap = None
                # 发送空的QPixmap而不是None
                self.magnifierUpdated.emit(QPixmap())
                return
                
            # 获取鼠标在图像中的位置
            img_pos = self.map_to_image(pos)
            x, y = int(img_pos.x()), int(img_pos.y())
            
            # 确保坐标在图像范围内
            h, w = self.cv_img.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                self.current_magnifier_pixmap = None
                # 发送空的QPixmap而不是None
                self.magnifierUpdated.emit(QPixmap())
                return
                
            # 计算放大区域的范围 - 根据橡皮擦大小动态调整
            eraser_radius = self.eraser_size // 2
            # 确保放大区域至少比橡皮擦大1.5倍，以便完整显示橡皮擦
            min_half_size = max(int(eraser_radius * 1.5), int(self.magnifier_size // (2 * self.magnifier_zoom)))
            half_size = min_half_size
            
            # 动态调整放大倍率，确保橡皮擦完整显示
            effective_zoom = self.magnifier_zoom
            if eraser_radius * 2 * self.magnifier_zoom > self.magnifier_size * 0.8:
                # 橡皮擦太大，需要降低放大倍率
                effective_zoom = self.magnifier_size * 0.8 / (eraser_radius * 2)
                # 增加显示区域以保持合适的视野
                half_size = int(eraser_radius * 1.2)
                print(f"[放大镜] 调整放大倍率为 {effective_zoom:.1f}，以适应橡皮擦大小 {self.eraser_size}")
            
            # 确保不超出图像边界，并强制转换为整数类型
            x1 = max(0, int(x - half_size))
            y1 = max(0, int(y - half_size))
            x2 = min(w, int(x + half_size))
            y2 = min(h, int(y + half_size))
            
            # 检查区域大小
            if x2 <= x1 or y2 <= y1:
                self.current_magnifier_pixmap = None
                # 发送空的QPixmap而不是None
                self.magnifierUpdated.emit(QPixmap())
                return  # 区域无效
                
            # 提取原始图像区域
            region = self.cv_img[y1:y2, x1:x2].copy()
            
            # 创建一个透明的覆盖层
            overlay = region.copy()
            
            # 在覆盖层上绘制橡皮擦轮廓 - 减小边框宽度
            center_x = int(x - x1)
            center_y = int(y - y1)
            eraser_radius = self.eraser_size // 2
            
            # 绘制橡皮擦轮廓（白色边框，改为1px宽）
            if self.eraser_shape_circle:
                cv2.circle(overlay, (center_x, center_y), eraser_radius, (255, 255, 255), 1)
            else:
                # 绘制方形橡皮擦轮廓
                top_left = (center_x - eraser_radius, center_y - eraser_radius)
                bottom_right = (center_x + eraser_radius, center_y + eraser_radius)
                cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), 1)
            
            # 如果正在编辑现有掩码，显示原始掩码区域
            if self.mask is not None and self.pending_mask_id is not None:
                try:
                    mask_region = self.mask[y1:y2, x1:x2].copy()
                    
                    # 根据当前模式确定颜色
                    color = (0, 255, 0, 128) if self.mode == "add" else (255, 0, 0, 128)
                    
                    # 创建一个半透明的颜色层
                    color_layer = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                    color_layer[mask_region] = color[:3]
                    
                    # 半透明混合
                    alpha = 0.3
                    overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
                except Exception as e:
                    print(f"[警告] 显示掩码区域出错: {e}")
            
            # 绘制十字线指示当前位置 - 使用更细的十字线
            if 0 <= center_x < overlay.shape[1] and 0 <= center_y < overlay.shape[0]:
                cv2.line(overlay, (center_x, 0), (center_x, overlay.shape[0]), (255, 255, 255), 1)
                cv2.line(overlay, (0, center_y), (overlay.shape[1], center_y), (255, 255, 255), 1)
            
            # 放大图像，使用动态调整的放大倍率
            magnified = cv2.resize(overlay, None, fx=effective_zoom, fy=effective_zoom, 
                                 interpolation=cv2.INTER_LINEAR)
            
            # 转换为 QPixmap
            rgb_img = cv2.cvtColor(magnified, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytesPerLine = ch * w
            qimg = QImage(rgb_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # 存储当前放大镜内容
            self.current_magnifier_pixmap = pixmap
            
            # 发送信号更新放大镜预览
            self.magnifierUpdated.emit(pixmap)
                
        except Exception as e:
            print(f"[错误] 更新放大镜时出错: {e}")
            import traceback
            print(traceback.format_exc())
            self.current_magnifier_pixmap = None
            try:
                # 发送空的QPixmap而不是None
                self.magnifierUpdated.emit(QPixmap())
            except:
                pass
