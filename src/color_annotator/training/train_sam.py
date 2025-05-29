import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from segment_anything import sam_model_registry
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class ClothingSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + '_mask.png')
        
        # 读取图像和掩码
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 转换为张量
        if self.transform:
            image = self.transform(image)
        
        # 归一化掩码
        mask = torch.from_numpy(mask).float() / 255.0
        
        return image, mask

class SAMFineTuner:
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="cuda"):
        self.device = device
        self.model_type = model_type
        
        # 初始化SAM模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def train(self, train_dir, val_dir, num_epochs=50, batch_size=4, learning_rate=1e-4):
        # 创建数据加载器
        train_dataset = ClothingSegmentationDataset(
            os.path.join(train_dir, "read_images"),
            os.path.join(train_dir, "masks"),
            transform=self.transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
        
        val_dataset = ClothingSegmentationDataset(
            os.path.join(val_dir, "read_images"),
            os.path.join(val_dir, "masks"),
            transform=self.transform
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.sam.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.sam.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for images, masks in train_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                mask_predictions = self.sam(images)
                loss = criterion(mask_predictions, masks)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.sam.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    mask_predictions = self.sam(images)
                    loss = criterion(mask_predictions, masks)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.sam.state_dict(), 
                         f'best_model_{self.model_type}.pth')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()

def main():
    # 设置参数
    model_type = "vit_b"
    # 使用已有的模型路径
    checkpoint_path = "src/color_annotator/checkpoints/sam_vit_b.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据集路径
    train_dir = "segmentation_dataset/train"
    val_dir = "segmentation_dataset/val"
    
    print(f"使用预训练模型: {checkpoint_path}")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")
    
    # 创建训练器并开始训练
    trainer = SAMFineTuner(model_type, checkpoint_path, device)
    trainer.train(train_dir, val_dir)

if __name__ == "__main__":
    main() 