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

class AISegmentationWidget(QWidget):
    segmentation_completed = pyqtSignal(object)  # 发送分割结果的信号
    
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
        try:
            # 如果有共享的SAM模型，则使用它
            if self.shared_sam_model is not None:
                print("[AI分割] 使用共享的SAM模型")
                # 创建SAMWrapper实例，使用共享的SAM模型
                self.model = SAMWrapper(self.shared_sam_model.sam)
                
                # 更新UI状态
                self.segment_btn.setEnabled(True)
                self.model_status_label.setText("模型状态: 使用共享模型")
                self.model_status_label.setStyleSheet("""
                    QLabel {
                        padding: 5px;
                        border-radius: 3px;
                        background-color: #d4edda;
                        color: #155724;
                    }
                """)
                
                return
                
            # 以下是原有的模型加载逻辑
            # 获取当前文件所在目录
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent.parent
            
            # 模型路径 - 使用src/color_annotator/checkpoints目录
            checkpoints_dir = project_root / "src" / "color_annotator" / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            
            # 检查是否需要从旧目录迁移模型文件
            old_checkpoints_dir = project_root / "checkpoints"
            if old_checkpoints_dir.exists():
                try:
                    # 迁移训练好的模型
                    old_model_path = old_checkpoints_dir / "best_sam_model.pth"
                    if old_model_path.exists():
                        import shutil
                        new_model_path = checkpoints_dir / "best_sam_model.pth"
                        if not new_model_path.exists():  # 只有在目标不存在时才复制
                            shutil.copy2(str(old_model_path), str(new_model_path))
                            QMessageBox.information(
                                self,
                                "迁移完成",
                                f"已将训练好的模型从\n{old_model_path}\n迁移到\n{new_model_path}"
                            )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "迁移警告",
                        f"迁移模型文件时出错：{str(e)}\n请手动将模型文件移动到{checkpoints_dir}"
                    )
            
            sam_checkpoint = checkpoints_dir / "sam_vit_b.pth"
            model_checkpoint = checkpoints_dir / "best_sam_model.pth"
            
            # 检查SAM基础模型
            if not sam_checkpoint.exists():
                reply = QMessageBox.question(
                    self,
                    "下载模型",
                    "SAM基础模型不存在，是否要下载？\n(大约需要下载600MB)",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    try:
                        import requests
                        import tqdm
                        
                        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                        response = requests.get(url, stream=True)
                        total_size = int(response.headers.get('content-length', 0))
                        
                        progress = QProgressDialog("正在下载SAM模型...", "取消", 0, total_size, self)
                        progress.setWindowModality(Qt.WindowModal)
                        
                        with open(sam_checkpoint, 'wb') as f:
                            for data in response.iter_content(chunk_size=1024):
                                if progress.wasCanceled():
                                    f.close()
                                    sam_checkpoint.unlink()
                                    return
                                f.write(data)
                                progress.setValue(f.tell())
                        
                        progress.close()
                    except Exception as e:
                        QMessageBox.critical(self, "下载失败", f"下载SAM模型失败：{str(e)}")
                        return
                else:
                    return
            
            # 检查训练好的模型
            self.using_base_model = False
            if not model_checkpoint.exists():
                reply = QMessageBox.question(
                    self,
                    "模型缺失",
                    "训练好的模型文件不存在，是否继续使用基础SAM模型？\n" +
                    "注意：基础模型可能无法准确分割服饰，建议使用训练好的模型。",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                self.using_base_model = True
            
            # 加载SAM模型
            sam_model = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
            self.model = SAMWrapper(sam_model)
            
            # 如果存在训练好的模型，则加载
            if model_checkpoint.exists():
                try:
                    # 设置安全加载选项
                    @contextmanager
                    def allow_numpy_scalar():
                        import numpy as np
                        _orig_scalar = np.core.multiarray.scalar
                        try:
                            yield
                        finally:
                            np.core.multiarray.scalar = _orig_scalar
                    
                    with allow_numpy_scalar():
                        checkpoint = torch.load(
                            model_checkpoint,
                            map_location=self.device,
                            weights_only=False  # 使用False以避免兼容性问题
                        )
                    
                    # 如果加载的是state_dict格式
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # 如果是直接的state_dict
                        self.model.load_state_dict(checkpoint)
                    
                    # 更新模型状态UI
                    self.model_status_label.setText("模型状态: 已加载训练模型")
                    self.model_status_label.setStyleSheet("""
                        QLabel {
                            padding: 5px;
                            border-radius: 3px;
                            background-color: #d4edda;
                            color: #155724;
                        }
                    """)
                except Exception as e:
                    reply = QMessageBox.question(
                        self,
                        "加载警告",
                        f"加载训练好的模型时出错：{str(e)}\n是否继续使用基础SAM模型？",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                    self.using_base_model = True
            
            self.model.to(self.device)
            self.model.eval()
            
            # 更新UI状态
            self.segment_btn.setEnabled(True)
            if self.using_base_model:
                self.segment_btn.setText("开始分割(基础模型)")
                self.segment_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #ffc107;
                        color: black;
                    }
                    QPushButton:hover {
                        background-color: #ffb300;
                    }
                """)
                self.model_status_label.setText("模型状态: 使用基础模型")
                self.model_status_label.setStyleSheet("""
                    QLabel {
                        padding: 5px;
                        border-radius: 3px;
                        background-color: #fff3cd;
                        color: #856404;
                    }
                """)
            else:
                self.segment_btn.setText("开始分割")
                self.segment_btn.setStyleSheet("")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型时出错：{str(e)}")
            self.segment_btn.setEnabled(False)
    
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
        """执行分割"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像！")
            return
        
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        
        if self.using_base_model:
            reply = QMessageBox.question(
                self,
                "确认",
                "当前使用的是基础SAM模型，可能无法准确分割服饰。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        try:
            print("[分割] 开始准备图像...")
            # 准备图像
            image = self.current_image.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            print(f"[分割] 图像尺寸: {image.shape}")
            
            # 图像预处理增强
            # 1. 应用双边滤波减少纹理影响，同时保留边缘
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 2. 增强对比度
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 确保图像尺寸合适
            orig_h, orig_w = image.shape[:2]
            if max(orig_h, orig_w) > 1024:
                scale = 1024 / max(orig_h, orig_w)
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                print(f"[分割] 调整图像尺寸至: {new_h}x{new_w}")
            
            print("[分割] 转换为PyTorch张量...")
            # 转换为PyTorch张量
            image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
            image = image / 255.0
            
            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            image = (image - mean) / std
            
            # 移动到设备
            image = image.to(self.device)
            print(f"[分割] 图像已加载到设备: {self.device}")
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            print("[分割] 执行模型推理...")
            # 执行分割
            with torch.no_grad():
                try:
                    outputs = self.model(image)
                    print(f"[分割] 模型输出尺寸: {outputs.shape}")
                    pred_mask = torch.sigmoid(outputs) > 0.35  # 降低阈值以获取更多细节
                    pred_mask = pred_mask.cpu().numpy()[0, 0]
                    print(f"[分割] 掩码尺寸: {pred_mask.shape}")
                except Exception as e:
                    print(f"[错误] 模型推理失败: {str(e)}")
                    raise
            
            print("[分割] 开始后处理...")
            # 后处理
            try:
                # 1. 标记连通区域
                print("[分割] 标记连通区域...")
                labeled_mask = measure.label(pred_mask)
                num_regions = np.max(labeled_mask)
                print(f"[分割] 找到 {num_regions} 个连通区域")
                
                regions = measure.regionprops(labeled_mask)
                print(f"[分割] 计算区域属性完成，共 {len(regions)} 个区域")
                
                if regions:
                    # 计算每个区域的特征
                    region_features = []
                    image_area = pred_mask.shape[0] * pred_mask.shape[1]
                    image_center = np.array([pred_mask.shape[0] / 2, pred_mask.shape[1] / 2])
                    
                    for region in regions:
                        # 跳过太小的区域
                        if region.area < image_area * 0.01:  # 增加最小面积阈值
                            continue
                            
                        # 基本特征
                        area_ratio = region.area / image_area
                        
                        # 正确计算宽高比
                        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
                        height = bbox[2] - bbox[0]
                        width = bbox[3] - bbox[1]
                        aspect_ratio = width / height if height > 0 else 0
                        
                        solidity = region.solidity  # 区域密实度
                        extent = region.extent     # 区域范围比
                        
                        # 位置特征
                        centroid = np.array(region.centroid)
                        dist_to_center = np.linalg.norm(centroid - image_center)
                        max_possible_dist = np.sqrt(image_center[0]**2 + image_center[1]**2)
                        center_score = 1 - (dist_to_center / max_possible_dist)
                        
                        # 形状特征
                        perimeter = region.perimeter
                        compactness = 4 * np.pi * region.area / (perimeter * perimeter + 1e-6)
                        
                        # 边缘特征
                        edge_pixels = region.perimeter_crofton
                        edge_density = edge_pixels / np.sqrt(region.area)
                        
                        # 计算总分
                        shape_score = (
                            (0.3 < aspect_ratio < 3.0) * 2.0 +  # 合理的宽高比
                            (solidity > 0.7) * 1.0 +            # 较高的密实度
                            (extent > 0.5) * 1.0 +              # 较大的范围比
                            (compactness > 0.3) * 1.0 +         # 适当的紧凑度
                            (edge_density < 0.5) * 1.0          # 边缘平滑度
                        )
                        
                        total_score = (
                            shape_score * 0.4 +                 # 形状特征权重
                            center_score * 0.3 +                # 中心位置权重
                            (0.1 < area_ratio < 0.8) * 0.3     # 合理的面积比例
                        )
                        
                        # 添加调试信息
                        print(f"[调试] 区域 {len(region_features)}: " + 
                              f"面积={region.area}, " +
                              f"宽高比={aspect_ratio:.2f}, " +
                              f"密实度={solidity:.2f}, " +
                              f"范围比={extent:.2f}, " +
                              f"中心得分={center_score:.2f}, " +
                              f"总分={total_score:.2f}")
                        
                        region_features.append({
                            'region': region,
                            'score': total_score,
                            'area': region.area,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'extent': extent,
                            'center_score': center_score,
                            'edge_density': edge_density
                        })
                    
                    if not region_features:
                        print("[警告] 没有找到有效的区域")
                        return None
                    
                    # 按总分排序，选择得分最高的区域
                    sorted_regions = sorted(region_features, key=lambda x: x['score'], reverse=True)
                    best_region = sorted_regions[0]
                    
                    print(f"[分割] 选择最佳区域 - " +
                          f"面积: {best_region['area']}, " +
                          f"占比: {best_region['area'] / image_area:.1%}, " +
                          f"宽高比: {best_region['aspect_ratio']:.2f}, " +
                          f"总分: {best_region['score']:.2f}")
                    
                    pred_mask = labeled_mask == best_region['region'].label
                    
                    # 验证选择的区域
                    if np.sum(pred_mask) < image_area * 0.05:  # 增加最小面积阈值
                        print("[警告] 选择的区域太小，可能无效")
                        return None
                
                # 2. 平滑边界
                print("[分割] 执行形态学操作...")
                # 先进行闭运算填充小孔
                kernel_close = np.ones((7,7), np.uint8)  # 增大核大小
                pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
                
                # 再进行开运算去除噪点
                kernel_open = np.ones((5,5), np.uint8)  # 增大核大小
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel_open)
                
                # 最后进行轻微的膨胀，确保边界完整
                kernel_dilate = np.ones((3,3), np.uint8)
                pred_mask = cv2.dilate(pred_mask, kernel_dilate, iterations=1)
                
                print("[分割] 形态学操作完成")
            
            except Exception as e:
                print(f"[警告] 后处理出错，使用原始掩码: {str(e)}")
                # 如果后处理失败，使用原始掩码继续
                pass
            
            # 3. 调整回原始尺寸
            print(f"[分割] 调整掩码尺寸至原始大小: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (self.current_image.shape[1], self.current_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 隐藏进度条
            self.progress_bar.setVisible(False)
            
            print("[分割] 检查结果...")
            # 安全检查
            if not isinstance(pred_mask, np.ndarray):
                raise ValueError("掩码不是有效的numpy数组")
            
            if pred_mask.shape != (self.current_image.shape[0], self.current_image.shape[1]):
                raise ValueError(f"掩码尺寸不匹配: 期望 {self.current_image.shape[:2]}, 实际 {pred_mask.shape}")
            
            # 确保掩码是布尔类型
            pred_mask = pred_mask.astype(bool)
            
            print(f"[分割] 发送结果... 掩码尺寸: {pred_mask.shape}, 类型: {pred_mask.dtype}")
            
            # 创建可视化图像
            vis_img = self.current_image.copy()
            vis_img[pred_mask] = [0, 255, 0]  # 将分割区域显示为绿色
            
            # 显示可视化结果
            h, w = vis_img.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(vis_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled_pixmap)
            
            # 发送结果
            self.segmentation_completed.emit(pred_mask)
            print("[分割] 完成")
            
            # 分析掩码区域的主要颜色
            print("[分析] 开始分析掩码区域的颜色...")
            img = self.current_image.copy()
            
            # 应用双边滤波减少纹理影响，同时保留边缘
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            masked_img = img.copy()
            masked_img[~pred_mask] = 0  # 将非掩码区域设为黑色
            
            # 使用颜色分析器分析主色，减少颜色数量，只提取最主要的颜色
            color_infos = self.color_analyzer.analyze_image_colors(masked_img, pred_mask, k=5)
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
                    color_mask = self.create_color_mask(masked_img, pred_mask, color_info.rgb)
                    
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
            print(f"[错误] 分割过程中出错：\n{traceback.format_exc()}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "错误", f"分割过程中出错：{str(e)}")

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