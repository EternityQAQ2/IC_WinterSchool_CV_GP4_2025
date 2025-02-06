import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------
# 1. 全局超参数设置
# ---------------------
TARGET_SIZE = (240, 240)  # 目标图像尺寸
NUM_EPOCHS = 20          # 训练轮数
BATCH_SIZE = 4           # 每批数据大小
LEARNING_RATE = 1e-4     # 学习率

# ---------------------
# 2. 数据增强/预处理
# ---------------------
train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 常用ImageNet统计值
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()  # 转换为Tensor，且通道顺序为 [C,H,W]
])

# 可视化示例（可选）
def visualize_images(fla_images, seg_images):
    """
    用于显示输入图像（fla_images）和分割图像（seg_images）.
    假设每个输入图像是一个包含三个通道的图像，分割图像是单通道图像.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(3):
        axes[0, i].imshow(fla_images[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Flair Image {i + 1}")
    for i in range(3):
        axes[1, i].imshow(seg_images[i].permute(1, 2, 0), cmap="gray")
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Segmentation {i + 1}")
    plt.tight_layout()
    plt.show()

# ---------------------
# 3. 自定义数据集
# ---------------------
class TumorDataSet(Dataset):
    """
    原始问题在于：一个文件夹下有多张切片，若把它们在 __getitem__ 中 stack 到一起，
    再经过 DataLoader 整合 batch，会导致 [batch_size, num_slices, channels, H, W] 的 5D 输入，
    与 2D 卷积要求的 4D 输入不匹配。

    为解决此问题，我们将所有文件夹下的所有切片「展开到同一个列表」中，
    使得一次只取一张切片 (fla, seg)，保证输入是 [C, H, W]，再批量合并成 [B, C, H, W]。
    """
    def __init__(self, fla_dir, seg_dir, transform=None):
        super(TumorDataSet, self).__init__()
        self.fla_dir = fla_dir
        self.seg_dir = seg_dir
        self.transform = transform

        # 收集所有 (fla_path, seg_path) 对
        self.fla_seg_pairs = []
        folder_ids = sorted(
            [f for f in os.listdir(fla_dir) if f.isdigit()],
            key=lambda x: int(x)
        )
        for folder_id in folder_ids:
            fla_path = os.path.join(fla_dir, folder_id)
            seg_path = os.path.join(seg_dir, folder_id)

            # 切片文件名列表
            fla_files = sorted(os.listdir(fla_path))
            seg_files = sorted(os.listdir(seg_path))

            for f_name, s_name in zip(fla_files, seg_files):
                self.fla_seg_pairs.append((
                    os.path.join(fla_path, f_name),
                    os.path.join(seg_path, s_name)
                ))

    def __len__(self):
        return len(self.fla_seg_pairs)

    def __getitem__(self, idx):
        fla_img_path, seg_img_path = self.fla_seg_pairs[idx]

        # 读取图像
        fla_img = cv2.imread(fla_img_path, cv2.IMREAD_COLOR)       # [H,W,3]
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)   # [H,W]

        if fla_img is None or seg_img is None:
            raise ValueError(f"Fail to load image/mask: {fla_img_path}, {seg_img_path}")

        # 二值化
        seg_img = (seg_img > 127).astype(np.uint8)

        # 调整图像尺寸到目标尺寸
        fla_img = cv2.resize(fla_img, TARGET_SIZE)
        seg_img = cv2.resize(seg_img, TARGET_SIZE)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=fla_img, mask=seg_img)
            fla_img = augmented['image']  # [3,H,W]
            seg_img = augmented['mask']   # [H,W]

        return fla_img, seg_img  # 返回单张切片: (C,H,W), (H,W)

# ---------------------
# 4. U-Net 网络结构
# ---------------------
class UnetCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UnetCNN, self).__init__()
        
        # ========== 编码器 ========== #
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 240 -> 120
        )
        
        # ========== 中间层 ========== #
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 120 -> 60
        )
        
        # ========== 解码器（新增第二次上采样） ========== #
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 60 -> 120
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 120 -> 240
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # ========== 下采样阶段 ========== #
        x = self.encoder(x)      # [B,64,120,120]
        x = self.middle(x)       # [B,128,60,60]
        
        # ========== 上采样阶段 ========== #
        x = self.up1(x)          # [B,64,120,120]
        x = self.conv1(x)        # [B,64,120,120]
        
        x = self.up2(x)          # [B,64,240,240]
        x = self.conv2(x)        # [B,64,240,240]
        
        x = self.out_conv(x)     # [B,1,240,240]
        return x


# ---------------------
# 5. DiceLoss 定义
# ---------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        """
        inputs: [B, 1, H, W] (模型输出, 已过Sigmoid)
        targets: [B, H, W] 或 [B, 1, H, W] (标签)
        """
        # 如果模型最后一层没有 Sigmoid，可以在这里先 sigmoid(inputs)
        # 但因为模型中已有 nn.Sigmoid()，此处可直接使用 inputs 即可。
        # 为稳妥，可以再一次保证处于(0,1):
        inputs = torch.clamp(inputs, 0, 1)

        # 若 targets 是 [B,H,W]，可添加一个通道维度
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (
            inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        return 1 - dice.mean()

# ---------------------
# 6. 训练主函数
# ---------------------
def main():
    model_save_path = "./model"
    os.makedirs(model_save_path, exist_ok=True)  # 创建模型保存目录
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建数据集与 DataLoader
    dataset = TumorDataSet(
        fla_dir="./train_data/fla", 
        seg_dir="./train_data/seg", 
        transform=train_transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 初始化模型、损失、优化器
    model = UnetCNN(in_channels=3, out_channels=1).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (fla_batch, seg_batch) in enumerate(dataloader):
            fla_batch = fla_batch.to(device)    # [B, 3, 240, 240]
            seg_batch = seg_batch.to(device)    # [B, 240, 240] 或 [B, 1, 240, 240]

            optimizer.zero_grad()
            outputs = model(fla_batch)          # [B, 1, 240, 240]
            loss = criterion(outputs, seg_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {epoch_loss / len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_save_path, "model.pth"))
    print("模型已保存！")

if __name__ == "__main__":
    main() 
