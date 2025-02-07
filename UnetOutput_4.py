
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ---------------------
# 1. 全局超参数设置
# ---------------------
TARGET_SIZE = (240, 240)  # 目标图像尺寸
NUM_EPOCHS = 20           # 训练轮数
BATCH_SIZE = 8            # 每批数据大小
LEARNING_RATE = 1e-3      # 初始学习率
KERNEL_SIZE = 3          # 卷积核大小
# ---------------------
# 2. 数据增强/预处理 COMPOSE流水线
# ---------------------
train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5), # 旋转？是否有必要？
    A.HorizontalFlip(p=0.5), # 水平翻转？是否有必要？
    A.RandomBrightnessContrast(p=0.2), # 提高亮度对比度
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 常用ImageNet统计值
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()  # 转换为Tensor，通道顺序为 [C, H, W]
])

# 验证集通常只做归一化，不做随机增强
val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 常用ImageNet统计值
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ---------------------
# 3. 自定义数据集 获取图像信息的输入信息的
# ---------------------
class TumorDataSet(Dataset):
    """
    将所有文件夹下的所有切片展开到同一个列表中，
    保证取出单张切片(fla, seg)，满足2D卷积输入要求 [C,H,W]。通道，高，宽。
    """
    def __init__(self, fla_dir, seg_dir, transform=None):
        super(TumorDataSet, self).__init__()
        self.fla_dir = fla_dir
        self.seg_dir = seg_dir
        self.transform = transform

        # 收集所有 (fla_path, seg_path) 对(地址对)
        self.fla_seg_pairs = []
        # 仅匹配纯数字命名的文件夹：1, 2, 3, ...
        folder_ids = sorted(
            [f for f in os.listdir(fla_dir) if f.isdigit()],
            key=lambda x: int(x)
        )
        # folder_id : 1, 2, 3, ...
        for folder_id in folder_ids:
            fla_path = os.path.join(fla_dir, folder_id)
            seg_path = os.path.join(seg_dir, folder_id)

            # 若对应的seg文件夹不存在，跳过
            if not os.path.isdir(fla_path) or not os.path.isdir(seg_path):
                continue

            # 切片文件名列表
            fla_files = sorted(os.listdir(fla_path)) # fla_1.png, fla_2.png, ...
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

        # 二值化：将原本的mask(0~255)转换成(0,1) 1代表肿瘤位置，0代表背景
        seg_img = (seg_img > 127).astype(np.uint8)

        # 调整图像尺寸到目标尺寸
        fla_img = cv2.resize(fla_img, TARGET_SIZE)
        seg_img = cv2.resize(seg_img, TARGET_SIZE)

        # 数据增强/预处理
        if self.transform:
            augmented = self.transform(image=fla_img, mask=seg_img)
            fla_img = augmented['image']  # [3,H,W]
            seg_img = augmented['mask']   # [H,W]

        return fla_img, seg_img  # 返回 (C,H,W), (H,W)

# ---------------------
# 4. 改进的 U-Net 网络结构（带跳跃连接）
# ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(out_channels), # 归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 编码器
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器（逆卷积）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        d1 = self.down1(x)  # [B,64,H,W] inchannel=3 -> outchannel=64
        p1 = self.maxpool(d1) # (2,2) stride = 2

        d2 = self.down2(p1) # [B,128,H/2,W/2] 64 -> 128
        p2 = self.maxpool(d2) 

        d3 = self.down3(p2) # [B,256,H/4,W/4] 128 -> 256
        p3 = self.maxpool(d3)

        d4 = self.down4(p3) # [B,512,H/8,W/8] 256 -> 512
        p4 = self.maxpool(d4)

        # 中间层
        bn = self.bottleneck(p4) # [B,1024,H/16,W/16] 512 -> 1024

        # 解码
        up_4 = self.up4(bn)                   # [B,512,H/8,W/8] 1024 -> 512
        cat_4 = torch.cat([up_4, d4], dim=1)  # skip connection# 跳跃连接,保留了一些低层特征
        c4 = self.conv4(cat_4)                # [B,512,H/8,W/8] 

        up_3 = self.up3(c4)                   # [B,256,H/4,W/4]
        cat_3 = torch.cat([up_3, d3], dim=1)
        c3 = self.conv3(cat_3)                # [B,256,H/4,W/4]

        up_2 = self.up2(c3)                   # [B,128,H/2,W/2]
        cat_2 = torch.cat([up_2, d2], dim=1)
        c2 = self.conv2(cat_2)                # [B,128,H/2,W/2]

        up_1 = self.up1(c2)                   # [B,64,H,W]
        cat_1 = torch.cat([up_1, d1], dim=1)
        c1 = self.conv1(cat_1)                # [B,64,H,W]

        out = self.out_conv(c1)               # [B,1,H,W]
        out = torch.sigmoid(out)              # 概率输出 [0,1]
        return out

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
        inputs = torch.clamp(inputs, 0, 1)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (
            inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        return 1 - dice.mean()

# ---------------------
# 6. 验证过程
# ---------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for fla_batch, seg_batch in dataloader:
            fla_batch = fla_batch.to(device)  # [B, 3, 240, 240]
            seg_batch = seg_batch.to(device)  # [B, 240, 240]

            outputs = model(fla_batch)        # [B, 1, 240, 240]
            loss = criterion(outputs, seg_batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ---------------------
# 7. 推理并保存带绿色轮廓的结果
# ---------------------
def inference_and_save_results(model, device, fla_dir, result_root):
    """
    对 fla_dir 下每个子文件夹的所有切片做推理，得到预测掩码后，
    将肿瘤区域用绿色线条(轮廓)勾勒在原图上，并保存到 result_root/{folder_id}/ 下。
    """
    model.eval()

    # 确保结果目录存在
    os.makedirs(result_root, exist_ok=True)

    # 排序获取子文件夹，如 1,2,3,...
    folder_ids = sorted(
        [f for f in os.listdir(fla_dir) if f.isdigit()],
        key=lambda x: int(x)
    )

    # 推理时的transform：只做归一化(保持与val_transform一致)
    infer_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    for folder_id in folder_ids:
        folder_path = os.path.join(fla_dir, folder_id)
        if not os.path.isdir(folder_path):
            continue

        # 创建对应的输出目录
        out_dir = os.path.join(result_root, folder_id)
        os.makedirs(out_dir, exist_ok=True)

        # 遍历该文件夹下的所有原图
        fla_files = sorted(os.listdir(folder_path))
        for f_name in fla_files:
            fla_path = os.path.join(folder_path, f_name)

            # 读取原图并预处理
            original_img = cv2.imread(fla_path, cv2.IMREAD_COLOR)
            if original_img is None:
                continue

            h, w, _ = original_img.shape
            resized_img = cv2.resize(original_img, TARGET_SIZE)  # 240x240

            # albumentations transform
            augmented = infer_transform(image=resized_img)
            inp_img = augmented['image'].unsqueeze(0).to(device)  # [1,3,240,240]

            # 模型推理
            with torch.no_grad():
                pred = model(inp_img)  # [1,1,240,240]
                pred_mask = pred.squeeze().cpu().numpy()  # [240,240]

            # 二值化
            pred_mask_bin = (pred_mask >= 0.5).astype(np.uint8)

            # 找轮廓（先将Mask映射回原图分辨率）
            mask_resized = cv2.resize(pred_mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 在原图上画绿线
            cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)

            # 保存结果
            save_path = os.path.join(out_dir, f_name)
            cv2.imwrite(save_path, original_img)
        print(f"Finished inference on folder: {folder_id}")

# ---------------------
# 8. 训练主函数（加动态学习率）
# ---------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建「训练集」和「验证集」
    train_dataset = TumorDataSet(
        fla_dir="./train_data/fla", 
        seg_dir="./train_data/seg", 
        transform=train_transform
    )
    val_dataset = TumorDataSet(
        fla_dir="./val_data/fla", 
        seg_dir="./val_data/seg", 
        transform=val_transform
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 初始化模型、损失、优化器
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 动态学习率调度器 (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # 目标是最小化验证集loss
        factor=0.5,      # 学习率缩放因子：若验证集loss不下降，则学习率 * 0.5
        patience=3,      # 若3轮验证集loss不下降，则触发一次缩放
        verbose=True,    # 打印学习率变化日志
        min_lr=1e-11     # 学习率下限
    )

    best_val_loss = float("inf")
    model_save_path = "./model"
    os.makedirs(model_save_path, exist_ok=True)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (fla_batch, seg_batch) in enumerate(train_loader):
            fla_batch = fla_batch.to(device)    # [B, 3, 240, 240]
            seg_batch = seg_batch.to(device)    # [B, 240, 240]

            optimizer.zero_grad()
            outputs = model(fla_batch)          # [B, 1, 240, 240]
            loss = criterion(outputs, seg_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每隔一定步数打印一次batch级别loss（你可根据需要调整频率）
            if (batch_idx + 1) % 10 == 0:
                print(f"Loss at batch {batch_idx+1}: {loss.item()}")

        # 计算训练集平均损失
        train_avg_loss = epoch_loss / len(train_loader)

        # 在验证集上评估
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_avg_loss:.4f}  Val Loss: {val_loss:.4f}")

        # 动态调整学习率
        scheduler.step(val_loss)

        # 如果验证损失更低，则保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            print("  -> Best model saved.")

    print("训练完成！")

    # 载入最佳权重（若已在训练中保存，可选择是否载入）
    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pth")))

    # 进行推理并把肿瘤区域用绿色线条标记保存
    # 下面演示对 train_data/fla 下的数据做推理并输出到 ./Results
    inference_and_save_results(model, device, fla_dir="./train_data/fla", result_root="./Results")
    print("推理完成，结果已输出到 ./Results/ 下！")

if __name__ == "__main__":
    main()
