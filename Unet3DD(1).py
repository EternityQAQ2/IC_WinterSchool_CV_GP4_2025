import os
import random
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm  # 进度条
import math

# ========== 参数设置 ==========
DATA_ROOT = "./dataset_segmentation/train"  
EPOCHS = 20             # 训练轮数
BATCH_SIZE = 4           
BASE_LR = 1e-4          # 初始学习率
VAL_RATIO = 0.2         # 验证集比例
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
OUT_DIR = "./output_3d"
os.makedirs(OUT_DIR, exist_ok=True)

# 假设之前保存的模型在此路径下
CHECKPOINT_PATH = "./models/0.74-1/best_model.pth"

torch.backends.cudnn.benchmark = True

# 保存参数信息
with open("parameters.txt", "w") as f:
    f.write(f"Device = {DEVICE}\n")
    f.write(f"EPOCHS = {EPOCHS}\n")
    f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
    f.write(f"BASE_LR = {BASE_LR}\n")
    f.write(f"VAL_RATIO = {VAL_RATIO}\n")

# ========== 数据预处理及增强 ==========
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

def random_flip_3d(img, seg):
    """随机沿深度维度翻转 (概率0.5)"""
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        seg = seg[:, :, ::-1]
    return img, seg

def random_rotate_3d(img, seg):
    """
    随机沿轴平面旋转 (0, 90, 180, 270度)
    注：只在 (H, W) 平面上旋转
    """
    k = random.randint(0, 3)
    img = np.rot90(img, k, axes=(0, 1))
    seg = np.rot90(seg, k, axes=(0, 1))
    return img, seg

def random_gamma_3d(img):
    """随机 gamma 调整 (0.8 ~ 1.2)"""
    g = random.uniform(0.8, 1.2)
    img = np.power(img, g)
    return img

class BrainTumorDataset3D(Dataset):
    def __init__(self, ids_list, root_dir=DATA_ROOT, mode='train'):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.ids = ids_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        pid_str = f"{pid:03d}" if isinstance(pid, int) else str(pid)
        fla_path = os.path.join(self.root_dir, pid_str, f"{pid_str}_fla.nii.gz")
        seg_path = os.path.join(self.root_dir, pid_str, f"{pid_str}_seg.nii.gz")

        fla_nii = nib.load(fla_path)
        seg_nii = nib.load(seg_path)
        fla_data = fla_nii.get_fdata(dtype=np.float32)  # [240,240,155]
        seg_data = seg_nii.get_fdata(dtype=np.float32)   # [240,240,155]

        # 归一化到 [0,1]
        fla_data /= (fla_data.max() + 1e-8)

        # 训练模式下做数据增强
        if self.mode == 'train':
            fla_data, seg_data = random_flip_3d(fla_data, seg_data)
            fla_data, seg_data = random_rotate_3d(fla_data, seg_data)
            fla_data = random_gamma_3d(fla_data)

        # pad到16的倍数 => [240,240,160]
        fla_data = pad_to_multiple_16(fla_data)
        seg_data = pad_to_multiple_16(seg_data)

        # 转为 [1, H, W, D]
        fla_tensor = torch.from_numpy(fla_data).unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_data).unsqueeze(0)
        return fla_tensor.float(), seg_tensor.float()

# ========== 网络结构（3D U-Net） ==========
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):  # dropout 调整为0.2
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

        # 注意：不再使用 sigmoid，直接返回 logits
        out = self.out_conv(c1)  # => [B,1,240,240,160]
        out_cropped = out[..., :155]  # 裁剪回 [B,1,240,240,155]
        return out_cropped

# ========== 损失函数 ==========
class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0):
        """
        结合 BCEWithLogitsLoss 和 Tversky-Focal Loss（使用连续值计算 Tversky 指数）
        调整参数为 alpha=0.5, beta=0.5, gamma=1.0 以与 Dice 指数更契合
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        # 将 logits 转换为概率
        pred_prob = torch.sigmoid(pred)
        tp = (pred_prob * target).sum(dim=(1, 2, 3, 4))
        fp = ((1 - target) * pred_prob).sum(dim=(1, 2, 3, 4))
        fn = (target * (1 - pred_prob)).sum(dim=(1, 2, 3, 4))
        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        focal_tversky = (1 - tversky) ** self.gamma
        focal_tversky = focal_tversky.mean()
        return bce_loss + focal_tversky

# ========== 训练和验证 ==========
def train_one_epoch(model, loader, optimizer, device, epoch, loss_fn):
    model.train()
    total_loss, total_dci = 0., 0.
    count = 0
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    loop = tqdm(loader, desc=f"[Train E{epoch}]", leave=False)
    for fla, seg in loop:
        fla, seg = fla.to(device), seg.to(device)
        optimizer.zero_grad()
        seg_cropped = seg[..., :155]  # 同步裁剪

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(fla)
                loss = loss_fn(out, seg_cropped)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(fla)
            loss = loss_fn(out, seg_cropped)
            loss.backward()
            optimizer.step()

        # 直接用模型输出计算训练指标（sigmoid后阈值设为0.5）
        pred_prob = torch.sigmoid(out)
        pred_bin = (pred_prob > 0.5).float()
        seg_bin = (seg_cropped > 0.5).float()
        inter = (pred_bin * seg_bin).sum().item()
        dci = 2 * inter / (pred_bin.sum() + seg_bin.sum() + 1e-6)

        total_loss += loss.item()
        total_dci += dci
        count += 1
        loop.set_postfix({"loss": f"{loss.item():.4f}", "DCI": f"{dci:.4f}"})
    return total_loss / count, total_dci / count

def val_one_epoch(model, loader, device, epoch, loss_fn):
    model.eval()
    total_loss = 0.0
    total_inter = 0.0
    total_pred_sum = 0.0
    total_seg_sum = 0.0
    count = 0
    loop = tqdm(loader, desc=f"[Val E{epoch}]", leave=False)
    with torch.no_grad():
        for fla, seg in loop:
            fla, seg = fla.to(device), seg.to(device)
            out = model(fla)
            seg_cropped = seg[..., :155]
            loss = loss_fn(out, seg_cropped)

            pred_prob = torch.sigmoid(out)
            pred_bin = (pred_prob > 0.5).float()
            seg_bin = (seg_cropped > 0.5).float()
            inter = (pred_bin * seg_bin).sum().item()
            pred_sum = pred_bin.sum().item()
            seg_sum = seg_bin.sum().item()

            total_inter += inter
            total_pred_sum += pred_sum
            total_seg_sum += seg_sum
            total_loss += loss.item()
            count += 1
            loop.set_postfix({"loss": f"{loss.item():.4f}"})
    dice = 2 * total_inter / (total_pred_sum + total_seg_sum + 1e-6)
    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss, dice

# ========== 主函数 ==========
def main():
    # 获取所有样本ID
    all_dirs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    all_ids = []
    for d in all_dirs:
        try:
            all_ids.append(int(d))
        except:
            all_ids.append(d)
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * VAL_RATIO)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Total: {len(all_ids)} train={len(train_ids)}, val={len(val_ids)}")

    train_set = BrainTumorDataset3D(train_ids, root_dir=DATA_ROOT, mode='train')
    val_set = BrainTumorDataset3D(val_ids, root_dir=DATA_ROOT, mode='val')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 构建模型、优化器、调度器和损失函数
    model = UNet3D().to(DEVICE)
    # 如果有多块GPU，则使用 DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 加载之前的模型checkpoint以继续训练
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-11)
    loss_fn = TverskyFocalLoss(alpha=0.5, beta=0.5, gamma=1.0)

    # 保存训练日志和学习率变化日志
    train_log_file = os.path.join(OUT_DIR, "training_log.txt")
    lr_log_file = os.path.join(OUT_DIR, "lr_changes.txt")
    with open(train_log_file, "w") as f:
        f.write("Epoch, TrainLoss, TrainDCI, ValLoss, ValDCI, LR\n")
    with open(lr_log_file, "w") as f:
        f.write("Epoch, New_LR\n")

    best_val_dci = -1.
    prev_lr = optimizer.param_groups[0]['lr']
    for epoch in range(1, EPOCHS + 1):
        trn_loss, trn_dci = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, loss_fn)
        val_loss, val_dci = val_one_epoch(model, val_loader, DEVICE, epoch, loss_fn)
        current_lr = optimizer.param_groups[0]['lr']
        log_str = (f"Epoch[{epoch}/{EPOCHS}] TrainLoss={trn_loss:.4f}, TrainDCI={trn_dci:.4f} | "
                   f"ValLoss={val_loss:.4f}, ValDCI={val_dci:.4f}, LR={current_lr:.6f}")
        print(log_str)
        with open(train_log_file, "a") as f:
            f.write(f"{epoch}, {trn_loss:.4f}, {trn_dci:.4f}, {val_loss:.4f}, {val_dci:.4f}, {current_lr:.6f}\n")

        # 调整学习率
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            with open(lr_log_file, "a") as f:
                f.write(f"{epoch}, {new_lr:.6f}\n")
            prev_lr = new_lr

        if val_dci > best_val_dci:
            best_val_dci = val_dci
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            print(f"  [*] Best updated! Val DCI={val_dci:.4f}")

    print("Finished training. Best Val DCI=%.4f" % best_val_dci)

    # 推理（对全部 val 集进行预测）
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    model.eval()
    print("Start inference on val set ...")
    for pid in val_ids:
        pid_str = f"{pid:03d}" if isinstance(pid, int) else str(pid)
        fla_path = os.path.join(DATA_ROOT, pid_str, f"{pid_str}_fla.nii.gz")
        if not os.path.exists(fla_path):
            continue
        nib_fla = nib.load(fla_path)
        fla_data = nib_fla.get_fdata(dtype=np.float32)
        fla_data /= (fla_data.max() + 1e-8)
        fla_data_pad = pad_to_multiple_16(fla_data)
        inp = torch.from_numpy(fla_data_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)  # => [1,1,240,240,155]
        pred_prob = torch.sigmoid(out)
        pred_bin = (pred_prob > 0.5).float().cpu().numpy()[0, 0]
        pred_nii = nib.Nifti1Image(pred_bin, nib_fla.affine)
        out_path = os.path.join(OUT_DIR, f"{pid_str}_pred.nii.gz")
        nib.save(pred_nii, out_path)
        print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
