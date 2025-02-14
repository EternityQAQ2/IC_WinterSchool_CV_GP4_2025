import os
import random
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条
import math

# ========== 参数设置 ==========
TEST_ROOT = "./test"                 # 测试集根目录，每个测试样本在 ./test/{id}
OUTPUT_ROOT = "./test_predictions_2"   # 测试结果输出根目录，每个结果保存在 ./test_predictions/{id}
CHECKPOINT_PATH = "./output_3d/best_model.pth"  # 训练好的模型路径
ANOTHER_PATH = "./models/0.74-1/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ========== 数据预处理函数 ==========
def pad_to_multiple_16(image_3d):
    """
    将深度维度补到16的倍数（例如155补到160）。
    """
    H, W, D = image_3d.shape
    needD = ((D + 15) // 16) * 16
    padD = needD - D
    if padD <= 0:
        return image_3d
    pad_block = np.zeros((H, W, padD), dtype=image_3d.dtype)
    padded = np.concatenate([image_3d, pad_block], axis=2)
    return padded

def crop_back_3d(pred_3d, original_depth=155):
    """
    将预测结果从 [240,240,160] 裁剪回 [240,240,155].
    """
    if pred_3d.shape[2] > original_depth:
        return pred_3d[..., :original_depth]
    return pred_3d

# ========== 网络结构（3D U-Net） ==========
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.down1 = ConvBlock3D(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = ConvBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = ConvBlock3D(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = ConvBlock3D(64, 128)
        self.pool4 = nn.MaxPool3d(2)
        
        self.bottleneck = ConvBlock3D(128, 256)
        
        # 解码器
        self.up4 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv4 = ConvBlock3D(256, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv3 = ConvBlock3D(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv2 = ConvBlock3D(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv1 = ConvBlock3D(32, 16)
        
        self.out_conv = nn.Conv3d(16, 1, 1)
        
    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        
        bn = self.bottleneck(p4)
        
        up4 = self.up4(bn)
        c4 = self.conv4(torch.cat([up4, d4], dim=1))
        up3 = self.up3(c4)
        c3 = self.conv3(torch.cat([up3, d3], dim=1))
        up2 = self.up2(c3)
        c2 = self.conv2(torch.cat([up2, d2], dim=1))
        up1 = self.up1(c2)
        c1 = self.conv1(torch.cat([up1, d1], dim=1))
        
        # 返回 logits（不使用 sigmoid）
        out = self.out_conv(c1)  # 输出尺寸 [B,1,240,240,160]
        out_cropped = out[..., :155]  # 裁剪回 [B,1,240,240,155]
        return out_cropped

# ========== 测试程序 ==========
def main():
    # 构建模型并加载训练好的权重
    model = UNet3D().to(DEVICE)
    # 如果有多块GPU，则使用 DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 加载 checkpoint
    if os.path.exists(ANOTHER_PATH):
        print(f"Loading checkpoint from {ANOTHER_PATH}")
        state_dict = torch.load(ANOTHER_PATH, map_location=DEVICE)
        # 如果模型保存时使用了 DataParallel，则需要去除 "module." 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        print("Checkpoint not found!")
        return

    model.eval()

    # 遍历测试集ID，假设测试ID从211到251
    for test_id in range(211, 252):
        test_id_str = f"{test_id:03d}"
        # 构造测试图像路径：假设格式为 ./test/{id}/{id}_fla.nii.gz
        test_dir = os.path.join(TEST_ROOT, test_id_str)
        test_file = os.path.join(test_dir, f"{test_id_str}_fla.nii.gz")
        if not os.path.exists(test_file):
            print(f"Test file not found for ID {test_id_str}")
            continue

        # 读取测试图像
        nib_img = nib.load(test_file)
        img_data = nib_img.get_fdata(dtype=np.float32)
        # 归一化
        img_data /= (img_data.max() + 1e-8)
        # pad 到16的倍数
        img_data_pad = pad_to_multiple_16(img_data)
        # 转换为张量，形状为 [1,1,H,W,D]
        input_tensor = torch.from_numpy(img_data_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(input_tensor)
            # 得到预测概率，并应用阈值
            pred_prob = torch.sigmoid(out)
            pred_bin = (pred_prob > 0.5).float().cpu().numpy()[0, 0]
            # 裁剪回原始深度
            pred_bin = crop_back_3d(pred_bin, original_depth=155)

        # 保存预测结果到 ./test_predictions/{id}/{id}_pred.nii.gz
        output_dir = os.path.join(OUTPUT_ROOT, test_id_str)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{test_id_str}_pred.nii.gz")
        pred_nii = nib.Nifti1Image(pred_bin, nib_img.affine)
        nib.save(pred_nii, output_file)
        print(f"[Saved] Prediction for ID {test_id_str} at {output_file}")

if __name__ == "__main__":
    main()
