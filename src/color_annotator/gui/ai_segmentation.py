import sys
from pathlib import Path
import torch
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QMessageBox, QProgressBar,
                            QComboBox, QCheckBox, QProgressDialog, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
from segment_anything import sam_model_registry
import torch.nn.functional as F
import os
import requests
import tqdm 
from contextlib import contextmanager
from skimage import morphology
from skimage import measure
import json

# 修改导入路径
import sys

from src.color_annotator.gui.image_viewer import ImageViewer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src.color_annotator.utils.color_analyzer import ColorAnalyzer

class ReferenceImageManager:
    def __init__(self, parent=None):
        self.parent = parent
        self.reference_images = {}
        self.current_image = None
        self.load_reference_images()
    
    def load_reference_images(self):
        """加载参考图像库"""
        reference_dir = Path("reference_images")
        if not reference_dir.exists():
            reference_dir.mkdir(exist_ok=True)
            return
        
        for img_path in reference_dir.glob("*.png"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.reference_images[img_path.stem] = {
                        'image': img,
                        'path': str(img_path)
                    }
            except Exception as e:
                print(f"加载参考图像失败 {img_path}: {e}")

    def add_reference_image(self, name, image):
        """添加参考图像"""
        self.reference_images[name] = {
            'image': image,
            'path': ""
        }
        self.current_image = image

class SAMWrapper(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
        self.sam.eval()
        
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
            
        # 使用更深的网络来提取更好的特征
        self.post_process = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(16, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1)
        )

    def forward(self, image):
        # 确保图像尺寸为1024x1024
        if image.shape[-2:] != (1024, 1024):
            image = F.interpolate(
                image,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )
        
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image)
            
            # 创建空的掩码提示
            target_mask = torch.zeros((image.shape[0], 1, 256, 256), device=image.device)
            
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=target_mask,
            )
            
            if dense_embeddings.shape[-2:] != (64, 64):
                dense_embeddings = F.interpolate(
                    dense_embeddings,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        masks = F.interpolate(
            low_res_masks,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        
        processed = self.post_process(masks)
        return processed

class ColorSegmentationModelWrapper:
    """Wrapper for the color segmentation model"""
    def __init__(self):
        self.model = None
        self.img_size = 512
        self.color_mapping = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Load the color segmentation model"""
        try:
            model_dir = Path(__file__).parent / "model"
            model_path = model_dir / "model.pt"
            color_mapping_path = model_dir / "color_mapping.json"
            
            if not model_path.exists():
                print(f"Color segmentation model not found at {model_path}")
                return
                
            if not color_mapping_path.exists():
                print(f"Color mapping not found at {color_mapping_path}")
                return
                
            # Load model with weights_only=False to ensure compatibility
            try:
                self.model = torch.jit.load(str(model_path), map_location=self.device)
            except Exception as e:
                print(f"Error loading model with default settings: {e}")
                print("Trying to load with weights_only=False...")
                self.model = torch.load(str(model_path), map_location=self.device, weights_only=False)
            
            self.model.eval()
            
            # Load color mapping
            with open(color_mapping_path, 'r') as f:
                self.color_mapping = json.load(f)
            
            print("Color segmentation model loaded successfully")
            
        except Exception as e:
            print(f"Error loading color segmentation model: {e}")
            self.model = None
    
    def segment_with_colors(self, image):
        """Segment the image and return masks for each color class"""
        if self.model is None:
            print("Color segmentation model not loaded")
            return None, None
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            if isinstance(image, np.ndarray):
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
        else:
            print("Invalid image format")
            return None, None
        
        # 增强预处理 - 使用双边滤波减少纹理但保留边缘
        img_filtered = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        
        # 增强对比度
        lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        img_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 计算边缘图用于后处理
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Resize image
        img_resized = cv2.resize(img_enhanced, (self.img_size, self.img_size))
        
        # Normalize image - use double precision
        img_float = img_resized.astype(np.float64) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float64)
        img_normalized = (img_float - mean) / std
        
        # Convert to tensor with double precision
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(torch.float64).to(self.device)
        
        try:
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                # 确保应用softmax获取合适的概率分布
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)[0].cpu().numpy()
            
            # Create color mask
            color_mask = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # Get class to color mapping
            class_to_color = {int(k): v for k, v in self.color_mapping['class_to_color'].items()}
            
            # Create mask for each color class
            masks = {}
            
            # 应用形态学操作进一步净化掩码
            kernel = np.ones((5, 5), np.uint8)
            
            for class_id, color in class_to_color.items():
                # Class ID in mask is class_id + 1 (0 is background)
                binary_mask = (pred == class_id + 1)
                
                # Skip empty masks or masks with very few pixels
                if np.sum(binary_mask) < 100:  # 增加最小像素阈值
                    continue
                
                # 应用形态学操作清理掩码，移除噪点
                cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
                
                # Skip if mask becomes empty after cleaning
                if np.sum(cleaned_mask) < 100:
                    continue
                
                # Add to color mask
                color_mask[cleaned_mask > 0] = color
                
                # Store mask
                masks[tuple(color)] = cleaned_mask > 0
            
            # Resize back to original size
            original_h, original_w = image.shape[:2]
            color_mask = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            edges_resized = cv2.resize(edges_dilated, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            # 后处理 - 考虑边缘信息提高边界准确性
            refined_masks = {}
            for color, mask in masks.items():
                resized_mask = cv2.resize((mask * 255).astype(np.uint8), 
                                         (original_w, original_h), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # 使用边缘信息细化掩码边界
                edge_mask = edges_resized > 0
                
                # 将边缘区域从掩码中移除，使得边界更加精确
                refined_mask = (resized_mask > 0) & ~edge_mask
                
                # 再次应用形态学操作，使掩码更平滑
                refined_mask = cv2.morphologyEx(refined_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                
                # 使用分水岭算法进一步细化区域
                distance = cv2.distanceTransform(refined_mask, cv2.DIST_L2, 3)
                ret, sure_fg = cv2.threshold(distance, 0.1*distance.max(), 255, 0)
                sure_fg = sure_fg.astype(np.uint8)
                
                refined_masks[color] = sure_fg > 0
            
            # 重建颜色掩码
            refined_color_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            for color, mask in refined_masks.items():
                refined_color_mask[mask] = color
            
            return refined_color_mask, refined_masks
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def segment_image(self, image):
        """Segment the image and return the color mask"""
        color_mask, _ = self.segment_with_colors(image)
        return color_mask

class AISegmentationWidget(QWidget):
    segmentation_completed = pyqtSignal(np.ndarray)  # 发送分割结果的信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.current_image = None
        self.reference_manager = ReferenceImageManager(self)
        self.using_base_model = False
        self.color_analyzer = ColorAnalyzer()
        
        # 共享SAM模型
        self.shared_sam_model = None
        
        # 初始化属性
        self.pending_annotation = None
        self.pending_mask_id = None
        
        # 初始化viewer
        self.viewer = ImageViewer()
        self.viewer.setFixedSize(1024, 660)
        
        # 初始化颜色分割模型
        self.color_model = None
        
        # 界面设置
        self.setupUI()
        self.loadModel()
    
    def setupUI(self):
        layout = QVBoxLayout()
        
        # 添加标题
        title_label = QLabel("AI分割工具")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 添加模型状态标签
        self.model_status_label = QLabel("模型状态: 未加载")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                border-radius: 3px;
                background-color: #f8d7da;
                color: #721c24;
            }
        """)
        layout.addWidget(self.model_status_label)
        
        # 参考图像控制区域
        ref_control_layout = QHBoxLayout()
        
        self.show_ref_checkbox = QCheckBox("显示参考图")
        self.show_ref_checkbox.setChecked(True)
        self.show_ref_checkbox.stateChanged.connect(self.toggle_reference_visibility)
        ref_control_layout.addWidget(self.show_ref_checkbox)
        
        self.ref_combo = QComboBox()
        self.ref_combo.currentIndexChanged.connect(self.change_reference_image)
        ref_control_layout.addWidget(self.ref_combo)
        
        add_ref_btn = QPushButton("添加参考图")
        add_ref_btn.clicked.connect(self.add_reference_image)
        ref_control_layout.addWidget(add_ref_btn)
        
        layout.addLayout(ref_control_layout)
        
        # 添加按钮
        button_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.loadModel)
        button_layout.addWidget(self.load_model_btn)
        
        self.segment_btn = QPushButton("开始分割")
        self.segment_btn.clicked.connect(self.performSegmentation)
        self.segment_btn.setEnabled(False)
        button_layout.addWidget(self.segment_btn)
        
        layout.addLayout(button_layout)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 添加参考图像预览区域
        self.reference_label = QLabel("参考图像")
        self.reference_label.setAlignment(Qt.AlignCenter)
        self.reference_label.setMinimumSize(200, 200)
        self.reference_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.reference_label)
        
        # 添加预览区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        layout.addWidget(self.preview_label)
        
        self.setLayout(layout)
        
        # 初始化参考图像下拉框
        self.update_reference_combo()
        
        # 信号连接
        if self.viewer:
            self.viewer.scaleChanged.connect(self.update_scale_ui)
            self.viewer.annotationAdded.connect(self.cache_annotation)
    
    def update_reference_combo(self):
        """更新参考图像下拉框"""
        self.ref_combo.clear()
        for name in self.reference_manager.reference_images.keys():
            self.ref_combo.addItem(name)
    
    def toggle_reference_visibility(self, state):
        """切换参考图像显示状态"""
        self.reference_label.setVisible(state == Qt.Checked)
        
    def change_reference_image(self, index):
        """切换参考图像"""
        if index >= 0:
            name = self.ref_combo.currentText()
            ref_data = self.reference_manager.reference_images.get(name)
            if ref_data:
                img = ref_data['image']
                if img is not None:
                    # 始终确保颜色空间是正确的BGR->RGB转换
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    h, w = img_rgb.shape[:2]
                    bytes_per_line = 3 * w
                    q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    # 更高质量的缩放
                    scaled_pixmap = pixmap.scaled(
                        self.reference_label.width() - 10,
                        self.reference_label.height() - 10,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.reference_label.setPixmap(scaled_pixmap)
    
    def add_reference_image(self):
        """添加参考图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                target_path = Path(file_path)
                
                # 读取图像，直接使用BGR格式保存，显示时再转换
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("无法读取图像")
                
                # 添加到参考图像管理器
                self.reference_manager.add_reference_image(target_path.stem, img)
                
                # 更新UI
                self.update_reference_combo()
                self.ref_combo.setCurrentText(target_path.stem)
                
                # 立即更新显示
                self.change_reference_image(self.ref_combo.currentIndex())
                
                QMessageBox.information(self, "成功", "参考图像添加成功！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"添加参考图像失败：{str(e)}")
    
    def loadModel(self):
        """Load the AI segmentation model"""
        try:
            if self.shared_sam_model is None:
                print("Initializing SAM model...")
                self.shared_sam_model = SAMWrapper()
            
            if self.shared_sam_model.sam_model is not None:
                self.model_status_label.setText("模型状态: 已加载")
                self.model_status_label.setStyleSheet("""
                    QLabel {
                        padding: 5px;
                        border-radius: 3px;
                        background-color: #d4edda;
                        color: #155724;
                    }
                """)
                self.segment_btn.setEnabled(True)
            else:
                print("Failed to load SAM model")
                self.model_status_label.setText("模型状态: 加载失败")
                self.model_status_label.setStyleSheet("""
                    QLabel {
                        padding: 5px;
                        border-radius: 3px;
                        background-color: #f8d7da;
                        color: #721c24;
                    }
                """)
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            self.model_status_label.setText(f"模型状态: 错误 - {str(e)}")
            self.model_status_label.setStyleSheet("""
                QLabel {
                    padding: 5px;
                    border-radius: 3px;
                    background-color: #f8d7da;
                    color: #721c24;
                }
            """)
        
        # Load color segmentation model
        try:
            print("Loading color segmentation model...")
            self.color_model = ColorSegmentationModelWrapper()
            if self.color_model.model is not None:
                self.model_status_label.setText("模型状态: 已加载")
                self.model_status_label.setStyleSheet("""
                    QLabel {
                        padding: 5px;
                        border-radius: 3px;
                        background-color: #d4edda;
                        color: #155724;
                    }
                """)
                self.segment_btn.setEnabled(True)
                print("Color segmentation model loaded successfully")
            else:
                print("Failed to load color segmentation model")
        except Exception as e:
            print(f"Error loading color segmentation model: {e}")
    
    def setImage(self, image):
        """设置要分割的图像"""
        self.current_image = image
        if image is not None:
            # 显示预览，确保颜色正确
            h, w = image.shape[:2]
            
            # 确保图像是RGB格式显示
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            
            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled_pixmap)
    
    def setViewer(self, viewer):
        """设置关联的ImageViewer"""
        self.viewer = viewer
    
    def performSegmentation(self):
        """Execute AI segmentation on the current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        try:
            # Create progress dialog
            progress = QProgressDialog("Performing segmentation...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("AI Segmentation")
            progress.show()
            progress.setValue(10)
            
            # Try to use the color model first, fall back to SAM if needed
            if self.color_model and self.color_model.model:
                # Use color segmentation model
                progress.setValue(30)
                
                # Segment image with color model
                color_mask, resized_masks = self.color_model.segment_with_colors(self.current_image)
                
                if color_mask is not None:
                    progress.setValue(50)
                    
                    # Create visualization
                    overlay = self.current_image.copy()
                    overlay[color_mask > 0] = color_mask[color_mask > 0]
                    
                    # Convert to QPixmap
                    h, w = overlay.shape[:2]
                    qimg = QImage(overlay.data, w, h, w * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    
                    # Display result
                    self.show_segmentation_preview(pixmap)
                    
                    # Emit segmentation completed signal with the color mask
                    self.segmentation_completed.emit(color_mask)
                    
                    progress.setValue(100)
                    progress.close()
                    return
                else:
                    # Fall back to SAM model if no colors found
                    print("No color segments found, falling back to SAM")
            
            # Use SAM model as fallback
            if self.shared_sam_model is None or self.shared_sam_model.sam_model is None:
                progress.close()
                QMessageBox.warning(self, "Model Not Loaded", "AI model not loaded properly.")
                return
            
            # Get current points
            fg_points = self.get_points_from_table(self.fg_table)
            bg_points = self.get_points_from_table(self.bg_table)
            
            if len(fg_points) == 0 and len(bg_points) == 0:
                progress.close()
                QMessageBox.warning(self, "No Points", "Please add at least one foreground or background point.")
                return
            
            progress.setValue(40)
            
            # Process points format for SAM
            input_point = []
            input_label = []
            
            for point in fg_points:
                input_point.append(point)
                input_label.append(1)  # 1 for foreground
                
            for point in bg_points:
                input_point.append(point)
                input_label.append(0)  # 0 for background
            
            # Convert to numpy arrays
            input_point = np.array(input_point)
            input_label = np.array(input_label)
            
            progress.setValue(60)
            
            # Run inference
            mask = self.shared_sam_model.forward(
                self.current_image,
                input_point,
                input_label
            )
            
            progress.setValue(80)
            
            if mask is None:
                progress.close()
                QMessageBox.warning(self, "Segmentation Failed", "Failed to generate mask.")
                return
            
            # Apply mask to image
            masked_img = self.current_image.copy()
            colored_mask = np.zeros_like(masked_img)
            colored_mask[mask > 0] = [0, 255, 0]  # Green mask
            
            # Create overlay
            alpha = 0.5
            cv2.addWeighted(colored_mask, alpha, masked_img, 1 - alpha, 0, masked_img)
            
            # Convert to QPixmap
            h, w, c = masked_img.shape
            qimg = QImage(masked_img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Display result
            self.show_segmentation_preview(pixmap)
            
            # Emit segmentation completed signal with the combined mask
            self.segmentation_completed.emit(mask)
            
            progress.setValue(100)
            progress.close()
        
        except Exception as e:
            import traceback
            progress.close()
            error_msg = f"Segmentation error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Segmentation Error", f"An error occurred during segmentation:\n{str(e)}")

    def create_color_mask(self, img, base_mask, target_color, tolerance=45):
        """创建特定颜色的掩码"""
        try:
            # 确保target_color是整数类型
            target_color = tuple(int(c) for c in target_color)
            
            # 转换为HSV色彩空间
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # 准备颜色数组
            target_color_array = np.uint8([[target_color]])
            target_hsv = cv2.cvtColor(target_color_array, cv2.COLOR_RGB2HSV)[0][0]
            
            # 创建HSV阈值范围，确保使用uint8类型
            lower_bound = np.array([
                max(0, int(target_hsv[0]) - tolerance),
                max(0, int(target_hsv[1]) - tolerance),
                max(0, int(target_hsv[2]) - tolerance)
            ], dtype=np.uint8)
            
            upper_bound = np.array([
                min(180, int(target_hsv[0]) + tolerance),
                min(255, int(target_hsv[1]) + tolerance),
                min(255, int(target_hsv[2]) + tolerance)
            ], dtype=np.uint8)
            
            # 创建颜色掩码
            color_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            
            # 与基础掩码相交
            final_mask = np.logical_and(color_mask, base_mask)
            
            # 应用形态学操作来清理掩码，增大核大小以合并相近区域
            kernel = np.ones((5,5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            return final_mask.astype(bool)
            
        except Exception as e:
            print(f"[错误] 创建颜色掩码时出错: {str(e)}")
            return None

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
                
            if mask.shape != (self.viewer.cv_img.shape[0], self.viewer.cv_img.shape[1]):
                print(f"[错误] 掩码尺寸不匹配: 期望 {self.viewer.cv_img.shape[:2]}, 实际 {mask.shape}")
                return

            # 分析掩码区域的主要颜色
            print("[分析] 开始分析掩码区域的颜色...")
            img = self.viewer.cv_img.copy()
            
            # 应用双边滤波减少纹理影响，同时保留边缘
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            masked_img = img.copy()
            masked_img[~mask] = 0  # 将非掩码区域设为黑色
            
            # 使用颜色分析器分析主色，减少颜色数量，只提取最主要的颜色
            color_infos = self.color_analyzer.analyze_image_colors(masked_img, mask, k=5)
            if not color_infos:
                print("[警告] 无法分析颜色")
                return
            
            # 只保留占比超过5%的主要颜色
            color_infos = [c for c in color_infos if c.percentage > 0.05]
            
            print(f"[分析] 检测到 {len(color_infos)} 种主要颜色:")
            for i, color_info in enumerate(color_infos):
                print(f"  {i+1}. RGB={color_info.rgb}, 占比={color_info.percentage:.1%}")
                
                try:
                    # 为每种主要颜色创建独立的掩码
                    color_mask = self.create_color_mask(masked_img, mask, color_info.rgb)
                    
                    # 应用额外的形态学操作来合并相似区域
                    kernel = np.ones((7,7), np.uint8)
                    color_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                    color_mask = color_mask.astype(bool)
                    
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
                    
                except Exception as e:
                    print(f"[警告] 处理颜色 {color_info.rgb} 时出错: {str(e)}")
                    continue
            
            # 更新显示
            print("[更新] 刷新预览...")
            self.update_annotation_preview()
            self.update_color_pie_chart()
            print("[完成] 分割结果处理完成")
            
        except Exception as e:
            import traceback
            print(f"[错误] 处理分割结果时出错：\n{traceback.format_exc()}")
            QMessageBox.critical(self, "错误", f"处理分割结果时出错：{str(e)}")

    def add_annotation_to_table(self, color_info, mask_id):
        """添加标注到主窗口的表格中"""
        if not hasattr(self, 'viewer') or not self.viewer:
            print("[错误] 未设置viewer引用")
            return
            
        try:
            # 获取主窗口引用
            main_window = self.parent()
            while main_window and not hasattr(main_window, 'annotation_table'):
                main_window = main_window.parent()
            
            if not main_window or not hasattr(main_window, 'annotation_table'):
                print("[错误] 未找到主窗口或表格")
                return
                
            table = main_window.annotation_table
            row = table.rowCount()
            table.insertRow(row)
            
            r, g, b = color_info.rgb
            percentage = color_info.percentage
            
            # 编号列
            id_item = QTableWidgetItem(str(row + 1))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setData(Qt.UserRole, mask_id)
            id_item.setData(Qt.UserRole + 1, (r, g, b))  # 存储RGB值
            table.setItem(row, 0, id_item)
            
            # 主色块
            color_container = QWidget()
            color_layout = QHBoxLayout(color_container)
            color_layout.setContentsMargins(0, 0, 0, 0)
            
            color_label = QLabel()
            color_label.setStyleSheet(f"""
                background-color: rgb({r}, {g}, {b});
                border: 1px solid #dee2e6;
                border-radius: 2px;
            """)
            color_label.setFixedSize(40, 40)  # 增大主色块
            
            color_layout.addStretch()
            color_layout.addWidget(color_label)
            color_layout.addStretch()
            
            table.setCellWidget(row, 1, color_container)
            
            # 占比列
            percentage_item = QTableWidgetItem(f"{percentage:.1%}")
            percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemIsEditable)
            percentage_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 2, percentage_item)
            
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
                    font-family: \"Segoe UI Symbol\";
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
            visibility_btn.setText("●")
            # 使用lambda捕获当前的mask_id
            visibility_btn.clicked.connect(lambda checked, mid=mask_id, r=row: main_window.toggle_mask_visibility(r, checked))
            
            visibility_widget = QWidget()
            visibility_layout = QHBoxLayout(visibility_widget)
            visibility_layout.addWidget(visibility_btn)
            visibility_layout.setAlignment(Qt.AlignCenter)
            visibility_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 3, visibility_widget)
            
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
            edit_btn.clicked.connect(lambda _, r=row, mid=mask_id: main_window.edit_annotation(r, mid))
            
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
            del_btn.clicked.connect(lambda _, mid=mask_id, r=row: main_window.delete_annotation(mid, r))
            
            op_layout.addWidget(edit_btn)
            op_layout.addWidget(del_btn)
            op_widget.setLayout(op_layout)
            table.setCellWidget(row, 4, op_widget)
            
            # 设置行高
            table.setRowHeight(row, 50)
            
            print(f"[表格] 添加标注记录：颜色={color_info.rgb}, 占比={percentage:.1%}, 掩码ID={mask_id}")
            
        except Exception as e:
            print(f"[错误] 添加标注到表格失败: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def update_annotation_preview(self):
        # 实现更新标注预览的逻辑
        pass

    def update_color_pie_chart(self):
        # 实现更新颜色饼图的逻辑
        pass

    def run_ai_segmentation(self):
        """执行AI自动分割"""
        if self.viewer.cv_img is None:
            QMessageBox.warning(self, "错误", "请先加载图像")
            return

        # 这里可以添加你的AI分割模型的调用逻辑
        # ...

        QMessageBox.information(self, "提示", "AI分割功能尚未实现")

    def run_segmentation(self):
        """执行基于点的人工标定分割"""
        self.viewer.run_sam_with_points()

    def update_scale_ui(self, scale):
        """更新缩放UI"""
        if hasattr(self, 'scale_slider'):
            slider_val = int(scale / self.viewer.base_scale * 100)
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(slider_val)
            self.scale_slider.blockSignals(False)
            if hasattr(self, 'scale_label'):
                self.scale_label.setText(f"{slider_val}%")

    def cache_annotation(self, color_and_mask):
        """缓存标注信息"""
        color, mask_id = color_and_mask
        self.pending_annotation = color
        self.pending_mask_id = mask_id

    def show_segmentation_preview(self, pixmap):
        """显示分割预览"""
        if hasattr(self, 'segmentation_preview_label'):
            w, h = pixmap.width(), pixmap.height()
            target_size = self.segmentation_preview_label.size()
            max_w, max_h = target_size.width(), target_size.height()

            if w / h > max_w / max_h:
                scaled = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)
            else:
                scaled = pixmap.scaledToHeight(max_h, Qt.SmoothTransformation)

            self.segmentation_preview_label.setPixmap(scaled)

    def segment_image_with_colors(self):
        """Segment the image with color model and show preview"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("正在分割图像...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)
            
            # Segment image with color model
            color_mask, resized_masks = self.color_model.segment_with_colors(self.current_image)
            
            if color_mask is not None and resized_masks:
                progress.setValue(50)
                
                # Apply overlay effect
                alpha = 0.7
                overlay = cv2.addWeighted(self.current_image, 1 - alpha, color_mask, alpha, 0)
                
                # Convert to QPixmap
                h, w = overlay.shape[:2]
                qimg = QImage(overlay.data, w, h, w * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Display result
                self.show_segmentation_preview(pixmap)
                
                # Convert color mask to binary mask for compatibility
                binary_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                # 检查是否有颜色像素，如果有则标记为1
                binary_mask[np.where((color_mask[:,:,0] > 0) | (color_mask[:,:,1] > 0) | (color_mask[:,:,2] > 0))] = 1
                
                print(f"生成的binary_mask的形状: {binary_mask.shape}, 类型: {binary_mask.dtype}")
                
                # Emit segmentation completed signal with binary mask
                self.segmentation_completed.emit(binary_mask)
                
                progress.setValue(100)
            else:
                progress.close()
                QMessageBox.warning(self, "错误", "颜色分割失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割过程出错: {str(e)}")
            import traceback
            traceback.print_exc() 