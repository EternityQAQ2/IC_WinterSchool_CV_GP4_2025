# 训练CNN模型 -3 
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
# 之前使用的是cv2的归一化，不可以用。
Transform_IC = transforms.Compose([
    transforms.ToPILImage(),   # OpenCV → PIL（自动 BGR→RGB）
    transforms.ToTensor(),     # 归一化到 [0,1]，并转换为 Tensor 张量 
])
def visualize_images(fla_images, seg_images):
    """
    用于显示输入图像（fla_images）和分割图像（seg_images）.
    假设每个输入图像是一个包含三个通道的图像，分割图像是单通道图像.
    """
    # 设置图像的子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 显示 fla_images（三个视图）
    for i in range(3):
        axes[0, i].imshow(fla_images[i].permute(1, 2, 0))  # permute 是为了从 [C, H, W] 转到 [H, W, C]
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Flair Image {i + 1}")

    # 显示 seg_images（三个分割图像）
    for i in range(3):
        axes[1, i].imshow(seg_images[i].permute(1, 2, 0), cmap="gray")  # 灰度图像
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Segmentation {i + 1}")

    plt.tight_layout()
    plt.show()


class TumorDataSet(Dataset):
    def __init__(self,fla_dir,seg_dir,transform=Transform_IC):
        self.fla_dir = fla_dir
        self.seg_dir = seg_dir
        self.transform = transform
        # 获取所有病人 ID（确保排序一致）
        self.folder_ids = sorted(os.listdir(fla_dir), key=lambda x: int(x))
    
    def __len__(self):
        return len(self.folder_ids) # 返回病人数量
    
    def __getitem__(self,idx):
        folder_id = self.folder_ids[idx] # 获取病人 ID
        fla_path = os.path.join(self.fla_dir, folder_id)
        seg_path = os.path.join(self.seg_dir, folder_id)
        # 读取大脑三视图
        fla_images_Idx = sorted(os.listdir(fla_path)) # fla_2_(XY).png 、fla_2_(XZ).png、fla_2_(YZ).png
        seg_images_Idx = sorted(os.listdir(seg_path)) # seg_2_(XY).png 、seg_2_(XZ).png、seg_2_(YZ).png

        fla_images = []
        for img_name in fla_images_Idx:
            img_path = os.path.join(fla_path, img_name)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR) # BGR 格式
            if self.transform:
                img = self.transform(img)  # 输入是 numpy 数组，自动触发 ToPILImage()
            fla_images.append(img) # 添加其中

        seg_images = []
        for img_name in seg_images_Idx:
            img_path = os.path.join(seg_path, img_name)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            if self.transform:
                img = self.transform(img)
            seg_images.append(img)

        return fla_images,seg_images # 返回大脑三视图和分割图
class UnetCNN(nn.Module):
    def __init__(self,in_channels=3,out_channels=1): 
        super(UnetCNN, self).__init__() 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)  # 输出通道数由任务决定
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    fla_dir = "./train_data/fla"
    seg_dir = "./train_data/seg"
    DatasetTrain = TumorDataSet(fla_dir, seg_dir)
    dataloader = torch.utils.data.DataLoader(DatasetTrain, batch_size=4, shuffle=True)

    # 测试代码
    # fla_images,seg_images = DatasetTrain.__getitem__(1)
    # 打印样本信息，确认数据加载是否正常
    # print(f"Number of flair images: {len(fla_images)}")
    # print(f"Number of segmentation images: {len(seg_images)}")
    # visualize_images(fla_images, seg_images)

    modeltest = UnetCNN(3,1)

    # 定义损失函数 & 优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失（适用于 Mask 预测）
    optimizer = optim.Adam(modeltest.parameters(), lr=0.001)

    # 训练 U-Net
    num_epochs = 10
    for epoch in range(num_epochs):
        for fla, seg in dataloader:
            optimizer.zero_grad()
            outputs = modeltest(fla)
            loss = criterion(outputs, seg)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")



if __name__ == "__main__":
    main()
         