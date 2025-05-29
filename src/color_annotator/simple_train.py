import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

# 定义一个更简单的U-Net模型用于服饰分割
class SimpleUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SimpleUNet, self).__init__()
        
        # 使用更强大的骨干网络 - ResNet18作为编码器
        try:
            import torchvision.models as models
            # 加载预训练的ResNet18
            if pretrained:
                print("使用预训练的ResNet18作为骨干网络...")
                resnet = models.resnet18(weights='DEFAULT')
            else:
                resnet = models.resnet18(weights=None)
                
            # 提取各个阶段的特征层
            self.firstconv = resnet.conv1
            self.firstbn = resnet.bn1
            self.firstrelu = resnet.relu
            self.firstmaxpool = resnet.maxpool
            self.encoder1 = resnet.layer1  # 64
            self.encoder2 = resnet.layer2  # 128
            self.encoder3 = resnet.layer3  # 256
            self.encoder4 = resnet.layer4  # 512
            
            # 解码器路径 - 上采样
            self.decoder4 = self._decoder_block(512, 256)
            self.decoder3 = self._decoder_block(256 + 256, 128)
            self.decoder2 = self._decoder_block(128 + 128, 64)
            self.decoder1 = self._decoder_block(64 + 64, 32)
            
            # 输出层
            self.final_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.final_relu1 = nn.ReLU(inplace=True)
            self.final_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.final_relu2 = nn.ReLU(inplace=True)
            self.final = nn.Conv2d(16, 1, kernel_size=1, bias=True)
            
            # 初始化偏置为较小的值，避免模型一开始就预测所有像素为前景
            self.final.bias.data.fill_(-1.0)
            
            # 注意力模块
            self.attention4 = self._attention_block(256, 256)
            self.attention3 = self._attention_block(128, 128)
            self.attention2 = self._attention_block(64, 64)
            self.attention1 = self._attention_block(32, 32)
            
        except ImportError:
            print("警告: 无法导入torchvision，回退到简单U-Net架构")
            # 编码器 (下采样)
            self.enc1 = self._encoder_block(3, 64)
            self.enc2 = self._encoder_block(64, 128)
            self.enc3 = self._encoder_block(128, 256)
            self.enc4 = self._encoder_block(256, 512)
            
            # 瓶颈层
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
            
            # 解码器 (上采样)
            self.dec4 = self._decoder_block(1024 + 512, 512)
            self.dec3 = self._decoder_block(512 + 256, 256)
            self.dec2 = self._decoder_block(256 + 128, 128)
            self.dec1 = self._decoder_block(128 + 64, 64)
            
            # 输出层
            self.final = nn.Conv2d(64, 1, kernel_size=1, bias=True)
            # 初始化偏置为较小的值，避免模型一开始就预测所有像素为前景
            self.final.bias.data.fill_(-1.0)
            
            # 注意力模块
            self.attention4 = self._attention_block(512, 512)
            self.attention3 = self._attention_block(256, 256)
            self.attention2 = self._attention_block(128, 128)
            self.attention1 = self._attention_block(64, 64)
    
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def _attention_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 检查是否使用ResNet骨干
        if hasattr(self, 'firstconv'):
            # ResNet编码器路径
            # 阶段1
            x = self.firstconv(x)
            x = self.firstbn(x)
            x = self.firstrelu(x)
            skip1 = x  # 保存第一个跳跃连接
            x = self.firstmaxpool(x)
            
            # 阶段2-5
            skip2 = self.encoder1(x)  # 保存第二个跳跃连接
            skip3 = self.encoder2(skip2)  # 保存第三个跳跃连接
            skip4 = self.encoder3(skip3)  # 保存第四个跳跃连接
            x = self.encoder4(skip4)  # 瓶颈特征
            
            # 解码器路径
            # 应用注意力机制
            x = self.decoder4(x)  # 上采样
            
            att4 = self.attention4(skip4)
            skip4 = skip4 * att4
            x = torch.cat([x, skip4], dim=1)  # 跳跃连接
            
            x = self.decoder3(x)  # 上采样
            
            att3 = self.attention3(skip3)
            skip3 = skip3 * att3
            x = torch.cat([x, skip3], dim=1)  # 跳跃连接
            
            x = self.decoder2(x)  # 上采样
            
            att2 = self.attention2(skip2)
            skip2 = skip2 * att2
            x = torch.cat([x, skip2], dim=1)  # 跳跃连接
            
            x = self.decoder1(x)  # 上采样
            
            att1 = self.attention1(x)
            x = x * att1
            
            # 最终输出层
            x = self.final_conv1(x)
            x = self.final_relu1(x)
            x = self.final_conv2(x)
            x = self.final_relu2(x)
            x = self.final(x)
        else:
            # 原始U-Net架构
            # 编码器路径
            enc1 = self.enc1[:-1](x)  # 不进行最后的池化
            enc1_pool = F.max_pool2d(enc1, kernel_size=2, stride=2)
            
            enc2 = self.enc2[:-1](enc1_pool)
            enc2_pool = F.max_pool2d(enc2, kernel_size=2, stride=2)
            
            enc3 = self.enc3[:-1](enc2_pool)
            enc3_pool = F.max_pool2d(enc3, kernel_size=2, stride=2)
            
            enc4 = self.enc4[:-1](enc3_pool)
            enc4_pool = F.max_pool2d(enc4, kernel_size=2, stride=2)
            
            # 瓶颈层
            bottleneck = self.bottleneck(enc4_pool)
            
            # 解码器路径
            # 应用注意力机制
            att4 = self.attention4(enc4)
            enc4 = enc4 * att4
            
            dec4 = self.dec4[:-1](torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
            
            att3 = self.attention3(enc3)
            enc3 = enc3 * att3
            
            dec3 = self.dec3[:-1](torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
            
            att2 = self.attention2(enc2)
            enc2 = enc2 * att2
            
            dec2 = self.dec2[:-1](torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
            
            att1 = self.attention1(enc1)
            enc1 = enc1 * att1
            
            dec1 = self.dec1[:-1](torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))
            
            # 输出层
            x = self.final(dec1)
        
        return x

# 数据集类
class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 直接从images子目录加载图像
        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            raise ValueError(f"图像目录不存在: {images_dir}")
            
        self.image_paths = []
        for ext in ['.jpg', '.JPG', '.png', '.tif']:
            self.image_paths.extend(list(images_dir.glob(f"*{ext}")))
            
        print(f"找到 {len(self.image_paths)} 张图像")
        
        # 检查标注目录 - 修正为正确的路径
        annotations_dir = self.data_dir / "annotations"
        if not annotations_dir.exists():
            print(f"警告: 标注目录不存在: {annotations_dir}")
        else:
            # 检查有多少图像有对应的标注
            annotated_count = 0
            for img_path in self.image_paths:
                ann_path = annotations_dir / f"{img_path.stem}.json"
                if ann_path.exists():
                    annotated_count += 1
            print(f"找到 {annotated_count}/{len(self.image_paths)} 张图像有对应的标注")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标注 - 修正为正确的路径
        ann_path = self.data_dir / "annotations" / f"{img_path.stem}.json"
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # 调试信息
                if idx == 0:  # 只打印第一个样本的信息
                    print(f"标注文件结构: {list(ann_data.keys())}")
                
                if 'masks' in ann_data:  # 处理新格式
                    for mask_id, mask_data in ann_data['masks'].items():
                        if 'size' not in mask_data or 'rle' not in mask_data:
                            continue
                            
                        mask_array = np.zeros(tuple(mask_data['size']), dtype=np.float32)
                        for start, length in mask_data['rle']:
                            if start + length <= len(mask_array.flat):
                                mask_array.flat[start:start + length] = 1
                            
                        if mask_array.shape != image.shape[:2]:
                            mask_array = cv2.resize(
                                mask_array,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        mask = np.maximum(mask, mask_array)
                        
                elif 'annotations' in ann_data:  # 处理旧格式
                    for ann in ann_data['annotations']:
                        if 'size' not in ann or 'rle' not in ann:
                            continue
                            
                        mask_array = np.zeros(tuple(ann['size']), dtype=np.float32)
                        for start, length in ann['rle']:
                            if start + length <= len(mask_array.flat):
                                mask_array.flat[start:start + length] = 1
                            
                        if mask_array.shape != image.shape[:2]:
                            mask_array = cv2.resize(
                                mask_array,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        mask = np.maximum(mask, mask_array)
                
                # 如果是第一个样本，打印掩码统计信息
                if idx == 0:
                    non_zero = np.count_nonzero(mask)
                    total = mask.size
                    print(f"掩码统计: 非零像素 {non_zero}/{total} ({non_zero/total:.2%})")
                    
            except Exception as e:
                print(f"处理标注文件出错 ({ann_path}): {e}")
        
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                print(f"数据增强错误 ({img_path}): {e}")
                # 如果增强失败，使用简单的调整大小
                image = cv2.resize(image, (256, 256))
                mask = cv2.resize(mask, (256, 256))
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).float().unsqueeze(0)
            
        # 确保掩码是单通道的
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

# 训练器类
class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 增强的数据增强策略
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.9),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.7),
                A.ChannelShuffle(p=0.2),
                A.ToGray(p=0.1),
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.MedianBlur(blur_limit=7, p=0.5),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def train(self, train_dir, val_dir=None, epochs=100, batch_size=8, 
              learning_rate=1e-4, weight_decay=2e-4, patience=15, 
              mixed_precision=True, save_dir='checkpoints'):
        """训练模型"""
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # 准备数据集
        train_dataset = ClothingDataset(train_dir, transform=self.train_transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        if val_dir:
            val_dataset = ClothingDataset(val_dir, transform=self.val_transform)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        
        # 优化器和学习率调度器 - 使用更高的学习率
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 使用余弦退火学习率调度器，加入预热阶段
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # 增大最大学习率
            total_steps=epochs * len(train_loader),
            pct_start=0.3,  # 增加预热阶段
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        # 损失函数
        bce_criterion = nn.BCEWithLogitsLoss()
        
        # 添加Dice损失函数
        def dice_loss(pred, target, smooth=1.0):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
            return 1 - dice
        
        # 添加Focal Loss
        def focal_loss(pred, target, alpha=0.8, gamma=2.0):
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = alpha * (1-pt)**gamma * bce
            return focal_loss.mean()
        
        # 添加Tversky Loss - 更好地处理类别不平衡
        def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1.0):
            pred = torch.sigmoid(pred)
            
            # 展平预测和目标
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            # 计算真阳性、假阳性和假阴性
            tp = (pred_flat * target_flat).sum()
            fp = ((1 - target_flat) * pred_flat).sum()
            fn = (target_flat * (1 - pred_flat)).sum()
            
            # 计算Tversky指数
            tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            
            return 1 - tversky
        
        # 添加边界感知损失
        def boundary_loss(pred, target, kernel_size=3):
            # 计算目标的边界
            target_boundaries = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - \
                               F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            
            # 计算预测的边界
            pred_sigmoid = torch.sigmoid(pred)
            pred_boundaries = F.max_pool2d(pred_sigmoid, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - \
                             F.max_pool2d(-pred_sigmoid, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            
            # 计算边界损失
            boundary_bce = F.binary_cross_entropy_with_logits(pred_boundaries, target_boundaries, reduction='mean')
            return boundary_bce
        
        # 训练记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': []
        }
        
        best_val_score = -float('inf')
        no_improve_epochs = 0
        
        # 使用混合精度训练
        from torch.amp import autocast, GradScaler
        amp_dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        scaler = GradScaler() if mixed_precision and self.device == 'cuda' else None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc="Training")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                if mixed_precision and scaler is not None:
                    with autocast(device_type=self.device, dtype=amp_dtype):
                        outputs = self.model(images)
                        
                        # 确保输出和目标大小一致
                        if outputs.shape != masks.shape:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                        # 组合损失函数
                        bce_loss_val = bce_criterion(outputs, masks)
                        dice_loss_val = dice_loss(outputs, masks)
                        focal_loss_val = focal_loss(outputs, masks)
                        tversky_loss_val = tversky_loss(outputs, masks)
                        boundary_loss_val = boundary_loss(outputs, masks)
                        
                        # 动态加权组合损失 - 根据训练进度调整权重
                        progress = epoch / epochs  # 训练进度比例
                        
                        # 早期阶段更注重BCE和边界损失，后期阶段更注重Dice和Tversky损失
                        bce_weight = 0.5 * (1 - progress) + 0.1 * progress
                        dice_weight = 0.2 * (1 - progress) + 0.4 * progress
                        focal_weight = 0.1
                        tversky_weight = 0.1 * (1 - progress) + 0.3 * progress
                        boundary_weight = 0.1 * (1 - progress) + 0.1 * progress
                        
                        loss = (bce_loss_val * bce_weight + 
                                dice_loss_val * dice_weight + 
                                focal_loss_val * focal_weight + 
                                tversky_loss_val * tversky_weight + 
                                boundary_loss_val * boundary_weight)
                    
                    scaler.scale(loss).backward()
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    
                    # 确保输出和目标大小一致
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                    
                    # 组合损失函数
                    bce_loss_val = bce_criterion(outputs, masks)
                    dice_loss_val = dice_loss(outputs, masks)
                    focal_loss_val = focal_loss(outputs, masks)
                    tversky_loss_val = tversky_loss(outputs, masks)
                    boundary_loss_val = boundary_loss(outputs, masks)
                    
                    # 动态加权组合损失 - 根据训练进度调整权重
                    progress = epoch / epochs  # 训练进度比例
                    
                    # 早期阶段更注重BCE和边界损失，后期阶段更注重Dice和Tversky损失
                    bce_weight = 0.5 * (1 - progress) + 0.1 * progress
                    dice_weight = 0.2 * (1 - progress) + 0.4 * progress
                    focal_weight = 0.1
                    tversky_weight = 0.1 * (1 - progress) + 0.3 * progress
                    boundary_weight = 0.1 * (1 - progress) + 0.1 * progress
                    
                    loss = (bce_loss_val * bce_weight + 
                            dice_loss_val * dice_weight + 
                            focal_loss_val * focal_weight + 
                            tversky_loss_val * tversky_weight + 
                            boundary_loss_val * boundary_weight)
                    
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()  # 每个批次更新学习率
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # 打印第一个批次的预测结果
                if epoch == 0 and len(train_losses) == 1:
                    with torch.no_grad():
                        pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                        print(f"第一个批次 - 预测掩码非零像素: {torch.count_nonzero(pred_masks).item()}/{pred_masks.numel()} ({torch.count_nonzero(pred_masks).item()/pred_masks.numel():.2%})")
                        print(f"第一个批次 - 真实掩码非零像素: {torch.count_nonzero(masks).item()}/{masks.numel()} ({torch.count_nonzero(masks).item()/masks.numel():.2%})")
                        print(f"第一个批次 - 输出值范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            if val_dir:
                val_metrics = self.evaluate(val_loader, mixed_precision)
                history['val_loss'].append(val_metrics['loss'])
                history['val_iou'].append(val_metrics['iou'])
                history['val_dice'].append(val_metrics['dice'])
                
                print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
                
                # 保存最佳模型
                val_score = val_metrics['iou'] * 0.6 + val_metrics['dice'] * 0.4
                if val_score > best_val_score:
                    best_val_score = val_score
                    # 保存最佳阈值
                    self.best_threshold = val_metrics.get('best_threshold', 0.5)
                    self.save_model(save_path / 'best_model.pth')
                    print(f"[保存] 发现更好的模型! Score: {val_score:.4f}, 最佳阈值: {self.best_threshold:.2f}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"[提示] 已经 {no_improve_epochs} 个周期没有改进")
            
            # 每个epoch保存一次检查点
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.save_model(save_path / f'checkpoint_epoch_{epoch+1}.pth')
                
            # 提前停止
            if no_improve_epochs >= patience:
                print(f"[提前停止] {patience} 个周期没有改进，停止训练")
                break
            
        # 保存训练历史
        self.save_history(history, save_path / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.plot_training_history(history, save_path / f'training_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        return history
    
    def evaluate(self, val_loader, mixed_precision=False):
        """评估模型"""
        self.model.eval()
        val_losses = []
        all_outputs = []
        all_masks = []
        bce_criterion = nn.BCEWithLogitsLoss()
        
        # 使用混合精度评估
        from torch.amp import autocast
        amp_dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        
        # 添加Dice损失函数
        def dice_loss(pred, target, smooth=1.0):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
            return 1 - dice
        
        # 添加Focal Loss
        def focal_loss(pred, target, alpha=0.8, gamma=2.0):
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = alpha * (1-pt)**gamma * bce
            return focal_loss.mean()
        
        # 添加Tversky Loss
        def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1.0):
            pred = torch.sigmoid(pred)
            
            # 展平预测和目标
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            # 计算真阳性、假阳性和假阴性
            tp = (pred_flat * target_flat).sum()
            fp = ((1 - target_flat) * pred_flat).sum()
            fn = (target_flat * (1 - pred_flat)).sum()
            
            # 计算Tversky指数
            tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            
            return 1 - tversky
        
        # 添加边界感知损失
        def boundary_loss(pred, target, kernel_size=3):
            # 计算目标的边界
            target_boundaries = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - \
                               F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            
            # 计算预测的边界
            pred_sigmoid = torch.sigmoid(pred)
            pred_boundaries = F.max_pool2d(pred_sigmoid, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - \
                             F.max_pool2d(-pred_sigmoid, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            
            # 计算边界损失
            boundary_bce = F.binary_cross_entropy_with_logits(pred_boundaries, target_boundaries, reduction='mean')
            return boundary_bce
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if mixed_precision and self.device == 'cuda':
                    with autocast(device_type=self.device, dtype=amp_dtype):
                        outputs = self.model(images)
                        
                        # 确保输出和目标大小一致
                        if outputs.shape != masks.shape:
                            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                            
                        # 组合损失函数
                        bce_loss_val = bce_criterion(outputs, masks)
                        dice_loss_val = dice_loss(outputs, masks)
                        focal_loss_val = focal_loss(outputs, masks)
                        tversky_loss_val = tversky_loss(outputs, masks)
                        boundary_loss_val = boundary_loss(outputs, masks)
                        
                        # 固定权重组合损失
                        loss = (bce_loss_val * 0.2 + 
                                dice_loss_val * 0.3 + 
                                focal_loss_val * 0.1 + 
                                tversky_loss_val * 0.3 + 
                                boundary_loss_val * 0.1)
                else:
                    outputs = self.model(images)
                    
                    # 确保输出和目标大小一致
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                    # 组合损失函数
                    bce_loss_val = bce_criterion(outputs, masks)
                    dice_loss_val = dice_loss(outputs, masks)
                    focal_loss_val = focal_loss(outputs, masks)
                    tversky_loss_val = tversky_loss(outputs, masks)
                    boundary_loss_val = boundary_loss(outputs, masks)
                    
                    # 固定权重组合损失
                    loss = (bce_loss_val * 0.2 + 
                            dice_loss_val * 0.3 + 
                            focal_loss_val * 0.1 + 
                            tversky_loss_val * 0.3 + 
                            boundary_loss_val * 0.1)
                
                val_losses.append(loss.item())
                
                # 保存输出和掩码用于后续阈值选择
                all_outputs.append(outputs.detach().cpu())
                all_masks.append(masks.detach().cpu())
                
                # 添加调试信息
                if i == 0:  # 只打印第一个批次的信息
                    print(f"验证集 - 输出形状: {outputs.shape}, 掩码形状: {masks.shape}")
                    print(f"验证集 - 输出值范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                    
                    # 使用固定阈值0.5查看预测结果
                    pred_masks_05 = (torch.sigmoid(outputs) > 0.5).float()
                    print(f"验证集 - 阈值0.5的预测掩码非零像素: {torch.count_nonzero(pred_masks_05).item()}/{pred_masks_05.numel()} ({torch.count_nonzero(pred_masks_05).item()/pred_masks_05.numel():.2%})")
                    print(f"验证集 - 真实掩码非零像素: {torch.count_nonzero(masks).item()}/{masks.numel()} ({torch.count_nonzero(masks).item()/masks.numel():.2%})")
        
        # 合并所有批次的输出和掩码
        all_outputs = torch.cat(all_outputs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # 将输出转换为概率
        all_probs = torch.sigmoid(all_outputs)
        
        # 尝试不同的阈值，找到最佳阈值
        best_iou = 0
        best_dice = 0
        best_threshold = 0.5
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            pred_masks = (all_probs > threshold).float()
            
            # 计算IoU和Dice
            intersection = torch.logical_and(pred_masks, all_masks).sum().float()
            union = torch.logical_or(pred_masks, all_masks).sum().float()
            iou = intersection / (union + 1e-6)
            
            dice = (2. * intersection) / (pred_masks.sum() + all_masks.sum() + 1e-6)
            
            print(f"阈值 {threshold:.1f} - IoU: {iou.item():.4f}, Dice: {dice.item():.4f}")
            
            # 更新最佳阈值
            if iou > best_iou:
                best_iou = iou
                best_dice = dice
                best_threshold = threshold
        
        print(f"最佳阈值: {best_threshold:.2f} - IoU: {best_iou:.4f}, Dice: {best_dice:.4f}")
        
        # 使用最佳阈值计算每个样本的指标
        pred_masks = (all_probs > best_threshold).float()
        
        # 计算每个样本的IoU和Dice
        ious = []
        dices = []
        for j in range(pred_masks.shape[0]):
            pred = pred_masks[j]
            target = all_masks[j]
            
            # 计算IoU
            intersection = torch.logical_and(pred, target).sum().float()
            union = torch.logical_or(pred, target).sum().float()
            iou = intersection / (union + 1e-6)
            ious.append(iou.item())
            
            # 计算Dice
            dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
            dices.append(dice.item())
        
        # 计算平均指标
        avg_iou = np.mean(ious) if ious else 0.0
        avg_dice = np.mean(dices) if dices else 0.0
        avg_loss = np.mean(val_losses) if val_losses else 0.0
        
        print(f"验证集 - 总体IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, Loss: {avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'best_threshold': best_threshold
        }
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_threshold': getattr(self, 'best_threshold', 0.5),
        }, path)
    
    def save_history(self, history, path):
        """保存训练历史"""
        with open(path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    def plot_training_history(self, history, path):
        """绘制训练历史图表"""
        plt.figure(figsize=(16, 6))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # IoU曲线
        if 'val_iou' in history and history['val_iou']:
            plt.subplot(1, 3, 2)
            plt.plot(history['val_iou'], label='Validation IoU', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.title('Validation IoU')
            plt.legend()
        
        # Dice曲线
        if 'val_dice' in history and history['val_dice']:
            plt.subplot(1, 3, 3)
            plt.plot(history['val_dice'], label='Validation Dice', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.title('Validation Dice')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="训练服饰分割模型 (简单U-Net)")
    parser.add_argument("--train-dir", type=str, default="segmentation_dataset/train", help="训练数据目录")
    parser.add_argument("--val-dir", type=str, default="segmentation_dataset/val", help="验证数据目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--mixed-precision", action="store_true", help="使用混合精度训练")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    
    args = parser.parse_args()
    
    # 检查数据目录
    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        raise ValueError(f"训练数据目录不存在: {args.train_dir}")
    
    if args.val_dir:
        val_dir = Path(args.val_dir)
        if not val_dir.exists():
            raise ValueError(f"验证数据目录不存在: {args.val_dir}")
    
    # 创建模型
    print("[初始化] 创建增强型U-Net模型...")
    model = SimpleUNet(pretrained=not args.no_pretrained)
    
    # 如果需要恢复训练
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[恢复训练] 从检查点加载: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建训练器
    print(f"[初始化] 创建训练器，使用设备: {args.device}")
    trainer = ModelTrainer(model, device=args.device)
    
    # 开始训练
    print("\n=== 开始训练 (增强型U-Net) ===")
    print(f"训练数据目录: {args.train_dir}")
    print(f"验证数据目录: {args.val_dir if args.val_dir else '无'}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"早停耐心值: {args.patience}")
    print(f"使用混合精度: {'是' if args.mixed_precision else '否'}")
    print(f"使用预训练权重: {'否' if args.no_pretrained else '是'}")
    print(f"模型保存目录: {args.save_dir}")
    
    history = trainer.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        mixed_precision=args.mixed_precision,
        save_dir=args.save_dir
    )
    
    print("\n[完成] 训练结束！")
    print(f"- 训练历史已保存在 {args.save_dir} 目录")
    print(f"- 训练曲线已保存在 {args.save_dir} 目录")
    print(f"- 最佳模型已保存为 {args.save_dir}/best_model.pth")

    # 打印最终结果
    if 'val_iou' in history and history['val_iou']:
        best_iou_idx = np.argmax(history['val_iou'])
        best_iou = history['val_iou'][best_iou_idx]
        best_dice = history['val_dice'][best_iou_idx]
        print(f"\n[最佳结果] Epoch {best_iou_idx + 1}")
        print(f"- IoU: {best_iou:.4f}")
        print(f"- Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main() 