import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
from segment_anything import sam_model_registry
import albumentations as A
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        """
        分割数据集类
        
        Args:
            img_dir: 图像目录
            annotation_dir: 标注目录（JSON文件）
            transform: 数据增强转换
        """
        self.img_dir = Path(img_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        
        # 获取所有标注文件
        self.annotations = []
        for json_path in self.annotation_dir.glob('*.json'):
            img_stem = json_path.stem
            # 支持多种图像格式
            img_formats = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF']
            img_path = None
            
            # 尝试所有支持的格式
            for fmt in img_formats:
                temp_path = self.img_dir / f"{img_stem}{fmt}"
                if temp_path.exists():
                    img_path = temp_path
                    break
            
            if img_path is not None:
                try:
                    # 尝试读取图像以确保它是有效的
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(json_path) as f:
                            annotation = json.load(f)
                            self.annotations.append((img_path, annotation))
                    else:
                        print(f"警告: 无法读取图像文件: {img_path}")
                except Exception as e:
                    print(f"警告: 处理图像文件时出错 {img_path}: {str(e)}")
            else:
                print(f"警告: 找不到对应的图像文件: {img_stem}.*")
        
        if not self.annotations:
            raise ValueError(f"在 {annotation_dir} 中没有找到有效的图像-标注对")
        
        print(f"找到 {len(self.annotations)} 个有效的图像-标注对")
        # 打印所有找到的图像路径
        print("\n找到的图像文件:")
        for img_path, _ in self.annotations:
            print(f"  - {img_path}")
    
    def __len__(self):
        return len(self.annotations)
    
    def rle_to_mask(self, rle_data, size):
        """将RLE格式转换为二进制掩码"""
        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
        current_pixel = 0
        
        for start, length in rle_data:
            mask[current_pixel:current_pixel + length] = 1
            current_pixel += length
        
        return mask.reshape(size)
    
    def __getitem__(self, idx):
        # 读取图像和标注
        img_path, annotation = self.annotations[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 随机选择一个标注
        mask_anno = np.random.choice(annotation['annotations'])
        
        # 将RLE转换为掩码
        mask = self.rle_to_mask(mask_anno['rle'], mask_anno['size'])
        mask = mask.astype(np.float32)
        
        # 记录原始尺寸
        original_size = image.shape[:2]
        
        # 调整图像到1024x1024（SAM需要这个尺寸）
        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 转换为张量
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float()
        
        return {
            'image': image,
            'mask': mask,
            'original_size': torch.tensor(original_size)
        }

def collate_fn(batch):
    """
    自定义的数据批处理函数
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    original_sizes = torch.stack([item['original_size'] for item in batch])
    
    return {
        'image': images,
        'mask': masks,
        'original_size': original_sizes
    }

class SAMWrapper(nn.Module):
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
        self.post_process = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            # 添加空洞卷积以获取更大的感受野
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )
        
        # 初始化为负偏置以减少过度分割
        for m in self.post_process.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, -1.0)

    def forward(self, batch):
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(batch['image'])
            
            target_mask = F.interpolate(
                batch['mask'].unsqueeze(1),
                size=(256, 256),
                mode='nearest'
            )
            
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
        
        # 添加L1正则化以促进稀疏性
        processed = self.post_process(masks)
        l1_reg = 0.1 * torch.mean(torch.abs(processed))
        self.l1_loss = l1_reg
        
        return processed

def calculate_metrics(pred_mask, true_mask):
    """
    计算分割评估指标
    
    Args:
        pred_mask: 预测掩码 (numpy array)
        true_mask: 真实掩码 (numpy array)
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 添加调试信息
    print("\n分割指标统计信息:")
    print(f"预测掩码 - 最小值: {pred_flat.min():.4f}, 最大值: {pred_flat.max():.4f}, 平均值: {pred_flat.mean():.4f}")
    print(f"真实掩码 - 最小值: {true_flat.min():.4f}, 最大值: {true_flat.max():.4f}, 平均值: {true_flat.mean():.4f}")
    print(f"预测掩码中正样本数量: {np.sum(pred_flat > 0)}")
    print(f"真实掩码中正样本数量: {np.sum(true_flat > 0)}")
    
    # 计算IoU
    intersection = np.logical_and(pred_flat, true_flat).sum()
    union = np.logical_or(pred_flat, true_flat).sum()
    iou = intersection / (union + 1e-6)
    
    # 计算Dice系数
    dice = (2.0 * intersection) / (pred_flat.sum() + true_flat.sum() + 1e-6)
    
    # 计算精确率、召回率和F1分数
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    
    # 打印详细的指标计算信息
    print(f"\n详细指标:")
    print(f"交集大小: {intersection}")
    print(f"并集大小: {union}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    Args:
        model: SAM模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        dict: 平均评估指标
    """
    model.eval()
    metrics_sum = {
        'iou': 0,
        'dice': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='评估')):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            target_mask = F.interpolate(
                batch['mask'].unsqueeze(1),
                size=(256, 256),
                mode='nearest'
            )
            
            outputs = model(batch)
            # 添加调试信息
            print(f"\n批次 {batch_idx + 1}:")
            print(f"输出张量统计 - 最小值: {outputs.min().item():.4f}, 最大值: {outputs.max().item():.4f}, 平均值: {outputs.mean().item():.4f}")
            
            # 使用动态阈值
            threshold = outputs.sigmoid().mean() * 0.7  # 使用平均值的70%作为阈值
            print(f"动态阈值: {threshold.item():.4f}")
            
            pred_masks = (torch.sigmoid(outputs) > threshold).cpu().numpy()
            true_masks = target_mask.cpu().numpy()
            
            # 计算每个样本的指标
            for i in range(pred_masks.shape[0]):
                print(f"\n样本 {i + 1}:")
                metrics = calculate_metrics(pred_masks[i, 0], true_masks[i, 0])
                for key in metrics:
                    metrics_sum[key] += metrics[key]
            total_samples += pred_masks.shape[0]
    
    # 计算平均指标
    metrics_avg = {k: v / total_samples for k, v in metrics_sum.items()}
    return metrics_avg

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """训练模型"""
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = Path('runs') / current_time
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard日志目录: {log_dir}")
    
    best_metrics = {
        'iou': 0,
        'dice': 0,
        'f1': 0
    }
    
    train_losses = []
    val_losses = []
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    for epoch in range(num_epochs):
        print(f'\n第 {epoch+1}/{num_epochs} 轮训练')
        
        model.train()
        epoch_train_loss = 0
        train_bar = tqdm(train_loader, desc='训练')
        
        for batch_idx, batch in enumerate(train_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            target_mask = F.interpolate(
                batch['mask'].unsqueeze(1),
                size=(256, 256),
                mode='nearest'
            )
            
            outputs = model(batch)
            
            # 计算每个像素的权重
            pos_weight = (target_mask == 0).float() * 0.1 + (target_mask == 1).float() * 2.0
            
            # 计算带权重的BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs, target_mask, 
                reduction='none'
            )
            weighted_loss = (bce_loss * pos_weight).mean()
            
            # 添加L1正则化损失
            total_loss = weighted_loss + model.l1_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 更激进的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            train_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})
            
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_step', total_loss.item(), global_step)
            
            if batch_idx % 100 == 0:
                pred_mask = (torch.sigmoid(outputs[0, 0]) > 0.5).cpu().numpy()
                true_mask = target_mask[0, 0].cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(pred_mask)
                ax1.set_title('预测掩码')
                ax2.imshow(true_mask)
                ax2.set_title('真实掩码')
                writer.add_figure('Masks/train', fig, global_step)
                plt.close(fig)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f'训练损失: {avg_train_loss:.4f}')
        
        if val_loader is not None:
            print("\n开始验证...")
            model.eval()
            metrics = evaluate_model(model, val_loader, device)
            
            for name, value in metrics.items():
                writer.add_scalar(f'Metrics/{name}', value, epoch)
                print(f'{name.upper()}: {value:.4f}')
            
            current_score = (metrics['iou'] + metrics['dice'] + metrics['f1']) / 3
            best_score = (best_metrics['iou'] + best_metrics['dice'] + best_metrics['f1']) / 3
            
            scheduler.step()
            
            if current_score > best_score:
                best_metrics = metrics
                model_save_path = Path('checkpoints') / 'best_sam_model.pth'
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                }, str(model_save_path))
                print(f'\n保存最佳模型到: {model_save_path}')
                print('最佳评估指标:')
                for name, value in metrics.items():
                    print(f'{name.upper()}: {value:.4f}')
    
    writer.close()
    return train_losses, val_losses, best_metrics

def main():
    """主函数"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent.parent
        dataset_dir = project_root / "segmentation_dataset"
        checkpoint_path = project_root / "src" / "color_annotator" / "checkpoints" / "sam_vit_b.pth"
        
        print(f"使用预训练模型: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到预训练模型: {checkpoint_path}")
        
        train_img_dir = dataset_dir / "train" / "read_images"
        train_anno_dir = dataset_dir / "train" / "annotations"
        
        val_img_dir = dataset_dir / "val" / "read_images"
        val_anno_dir = dataset_dir / "val" / "annotations"
        
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.2),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = SegmentationDataset(train_img_dir, train_anno_dir, train_transform)
        print(f"训练集大小: {len(train_dataset)} 样本")
        
        val_dataset = None
        val_loader = None
        
        if val_img_dir.exists() and val_anno_dir.exists() and len(list(val_img_dir.glob('*'))) > 0:
            val_dataset = SegmentationDataset(val_img_dir, val_anno_dir, val_transform)
            print(f"验证集大小: {len(val_dataset)} 样本")
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
        
        sam_model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        model = SAMWrapper(sam_model)
        model = model.to(device)
        
        # 使用基础的BCE损失函数，权重在训练循环中动态计算
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # 使用不同的学习率
        optimizer = optim.AdamW([
            {'params': model.sam.mask_decoder.parameters(), 'lr': 5e-5},
            {'params': model.post_process.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
        
        num_epochs = 3
        
        train_losses, val_losses, best_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device
        )
        
        print('\n训练完成！')
        print('\n最终评估指标:')
        for name, value in best_metrics.items():
            print(f'{name.upper()}: {value:.4f}')
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main() 