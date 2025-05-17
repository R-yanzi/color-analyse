# train_color_model.py
# 用于训练主色提取 CNN 模型（输入图像块，输出 RGB）

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 参数
DATA_PATH = "data/dataset_rgb_region.npz"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "color_model.pt"
DEVICE = torch.device("cpu")

# 自定义数据集
class ColorDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.images = data["images"].astype(np.float32) / 255.0  # 归一化
        self.labels = data["labels"].astype(np.float32) / 255.0  # RGB归一化

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx]).permute(2, 0, 1)  # HWC → CHW
        y = torch.tensor(self.labels[idx])
        return x, y

# 简单 CNN 模型
class ColorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 3),  # 输出 RGB
            nn.Sigmoid()       # 保证输出在 0~1 范围内
        )

    def forward(self, x):
        return self.net(x)

# 加载数据
full_dataset = ColorDataset(DATA_PATH)
split = int(0.8 * len(full_dataset))
train_set, val_set = torch.utils.data.random_split(full_dataset, [split, len(full_dataset) - split])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# 初始化模型
model = ColorCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练过程
train_loss_log, val_loss_log = [], []
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)
    train_loss_log.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)

    val_loss /= len(val_loader.dataset)
    val_loss_log.append(val_loss)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[保存] 模型已保存至 {MODEL_SAVE_PATH}")

# 绘图
plt.plot(train_loss_log, label="Train")
plt.plot(val_loss_log, label="Val")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Loss Curve")
plt.grid()
plt.savefig("loss_curve.png")
plt.show()
