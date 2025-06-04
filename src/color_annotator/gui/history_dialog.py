from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QMessageBox, QFrame, QSplitter,
    QWidget, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
from datetime import datetime

class HistoryDialog(QDialog):
    """历史记录对话框，用于显示和管理历史记录"""
    
    # 定义信号
    historySelected = pyqtSignal(str)  # 选择历史记录时发出信号，参数为快照ID
    
    def __init__(self, history_manager, parent=None):
        super().__init__(parent)
        self.history_manager = history_manager
        self.selected_snapshot_id = None
        
        self.setWindowTitle("标定历史记录")
        self.resize(800, 600)
        
        # 创建布局
        main_layout = QHBoxLayout(self)
        
        # 左侧历史列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.history_list = QListWidget()
        self.history_list.setMinimumWidth(300)
        self.history_list.currentItemChanged.connect(self.on_history_item_selected)
        
        left_layout.addWidget(QLabel("历史记录列表:"))
        left_layout.addWidget(self.history_list)
        
        # 右侧预览区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_scroll.setWidget(self.preview_container)
        
        self.info_label = QLabel("选择左侧历史记录查看详情")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 14px; color: #666;")
        
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumSize(300, 300)
        self.preview_image.setStyleSheet("border: 1px solid #ccc;")
        
        self.preview_layout.addWidget(self.info_label)
        self.preview_layout.addWidget(self.preview_image)
        self.preview_layout.addStretch()
        
        right_layout.addWidget(QLabel("预览:"))
        right_layout.addWidget(self.preview_scroll)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.restore_btn = QPushButton("恢复到此状态")
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self.restore_selected)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.restore_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # 添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter)
        
        # 添加底部按钮
        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.create_separator())
        bottom_layout.addLayout(button_layout)
        
        # 组合布局
        container_layout = QVBoxLayout()
        container_layout.addWidget(splitter)
        container_layout.addLayout(bottom_layout)
        
        self.setLayout(container_layout)
        
        # 加载历史记录
        self.load_history_list()
        
    def load_history_list(self):
        """加载历史记录列表"""
        self.history_list.clear()
        
        history_entries = self.history_manager.get_history_list()
        if not history_entries:
            item = QListWidgetItem("没有历史记录")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.history_list.addItem(item)
            return
            
        for entry in history_entries:
            timestamp = entry.get("timestamp", 0)
            date_str = entry.get("date", "未知时间")
            description = entry.get("description", "标定快照")
            
            item = QListWidgetItem(f"{date_str} - {description}")
            item.setData(Qt.UserRole, entry.get("id"))
            self.history_list.addItem(item)
            
    def on_history_item_selected(self, current, previous):
        """当选择历史记录项时触发"""
        if not current:
            self.selected_snapshot_id = None
            self.restore_btn.setEnabled(False)
            self.info_label.setText("选择左侧历史记录查看详情")
            self.preview_image.clear()
            return
            
        snapshot_id = current.data(Qt.UserRole)
        if not snapshot_id:
            return
            
        self.selected_snapshot_id = snapshot_id
        self.restore_btn.setEnabled(True)
        
        # 查找对应的历史记录
        history_entries = self.history_manager.get_history_list()
        entry = next((e for e in history_entries if e.get("id") == snapshot_id), None)
        
        if not entry:
            self.info_label.setText("无法加载历史记录详情")
            return
            
        # 显示历史记录信息
        date_str = entry.get("date", "未知时间")
        description = entry.get("description", "标定快照")
        mask_count = len(entry.get("masks", []))
        
        info_text = f"时间: {date_str}\n描述: {description}\n掩码数量: {mask_count}"
        self.info_label.setText(info_text)
        
        # 加载预览图
        self.load_preview_image(snapshot_id)
        
    def load_preview_image(self, snapshot_id):
        """加载预览图像"""
        try:
            # 恢复掩码
            masks = self.history_manager.restore_snapshot(snapshot_id)
            if not masks:
                self.preview_image.setText("无法加载预览图")
                return
                
            # 获取原始图像
            parent = self.parent()
            if parent and hasattr(parent, "viewer") and parent.viewer.cv_img is not None:
                img = parent.viewer.cv_img.copy()
                
                # 创建预览图像
                overlay = np.zeros_like(img, dtype=np.uint8)
                
                # 绘制所有掩码
                for entry in masks.values():
                    if not entry.get("visible", True):
                        continue
                        
                    mask = entry.get("mask")
                    if mask is None:
                        continue
                        
                    color = entry.get("color", (0, 255, 0))
                    r, g, b = color
                    
                    # 确保掩码形状正确
                    if mask.shape != overlay.shape[:2]:
                        continue
                        
                    overlay[mask] = (b, g, r)
                
                # 叠加时只让掩码部分半透明，其余部分完全显示原图
                alpha = 0.7
                mask_area = (overlay > 0).any(axis=2)
                
                preview = img.copy()
                preview[mask_area] = cv2.addWeighted(img[mask_area], 1 - alpha, overlay[mask_area], alpha, 0)
                
                # 转换为QPixmap并显示
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                h, w, _ = preview_rgb.shape
                qimg = QImage(preview_rgb.data, w, h, w * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # 调整大小以适应预览区域
                self.preview_image.setPixmap(pixmap.scaled(
                    self.preview_image.width(), 
                    self.preview_image.height(),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            else:
                self.preview_image.setText("无法加载原始图像")
        except Exception as e:
            self.preview_image.setText(f"加载预览图失败: {str(e)}")
            
    def restore_selected(self):
        """恢复选中的历史记录"""
        if not self.selected_snapshot_id:
            return
            
        reply = QMessageBox.question(
            self,
            "确认恢复",
            "确定要恢复到此历史状态吗？当前未保存的修改将丢失。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.historySelected.emit(self.selected_snapshot_id)
            self.accept()
            
    def create_separator(self):
        """创建分隔线"""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator 