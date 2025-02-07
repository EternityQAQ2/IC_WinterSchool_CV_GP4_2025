
# # 训练CNN模型 -4
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# # 定义统一的目标尺寸
# TARGET_SIZE = (240, 240)  # 根据需求调整

# # 之前使用的是cv2的归一化，不可以用。
# Transform_IC = transforms.Compose([
#     transforms.ToPILImage(),   # OpenCV → PIL（自动 BGR→RGB）
#     transforms.Resize(TARGET_SIZE, interpolation=Image.NEAREST),  # 最近邻插值
#     transforms.CenterCrop(TARGET_SIZE),  # 中心裁剪（可选）
#     transforms.ToTensor(),     # 归一化到 [0,1]，并转换为 Tensor 张量 
# ])


# def visualize_images(fla_images, seg_images):
#     """
#     用于显示输入图像（fla_images）和分割图像（seg_images）.
#     假设每个输入图像是一个包含三个通道的图像，分割图像是单通道图像.
#     """
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     for i in range(3):
#         axes[0, i].imshow(fla_images[i].permute(1, 2, 0))
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"Flair Image {i + 1}")
#     for i in range(3):
#         axes[1, i].imshow(seg_images[i].permute(1, 2, 0), cmap="gray")
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"Segmentation {i + 1}")
#     plt.tight_layout()
#     plt.show()

# class TumorDataSet(Dataset):
#     def __init__(self, fla_dir, seg_dir, fla_transform=Transform_IC):
#         self.fla_dir = fla_dir
#         self.seg_dir = seg_dir
#         self.fla_transform = fla_transform
#         # 按数字排序病人文件夹
#         self.folder_ids = sorted(
#             [f for f in os.listdir(fla_dir) if f.isdigit()],
#             key=lambda x: int(x)
#         )
    
#     def __len__(self):
#         return len(self.folder_ids)
    
#     def __getitem__(self, idx):
#         folder_id = self.folder_ids[idx]
#         fla_path = os.path.join(self.fla_dir, folder_id)
#         seg_path = os.path.join(self.seg_dir, folder_id)

#         # 读取并排序文件名（保持原排序）
#         fla_images_Idx = sorted(os.listdir(fla_path))
#         seg_images_Idx = sorted(os.listdir(seg_path))

#         # 读取并调整尺寸
#         fla_images = []
#         seg_images = []
#         for fla_name, seg_name in zip(fla_images_Idx, seg_images_Idx):
#             # 处理 FLA 图像
#             fla_img = cv2.imread(os.path.join(fla_path, fla_name), cv2.IMREAD_COLOR)
#             fla_img = self.fla_transform(fla_img)
#             fla_images.append(fla_img)
            
#             # 处理 SEG 标签
#             seg_img = cv2.imread(os.path.join(seg_path, seg_name), cv2.IMREAD_GRAYSCALE)
#             seg_img = (seg_img > 127).astype(np.uint8)  # 二值化
#             seg_img = self.fla_transform(seg_img)
#             seg_images.append(seg_img)

#         return torch.stack(fla_images), torch.stack(seg_images)

# class UnetCNN(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(UnetCNN, self).__init__()
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, out_channels, kernel_size=1),
#             nn.Sigmoid()  # 输出概率图
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# def main():
#     model_save_path = "./model"
#     os.makedirs(model_save_path, exist_ok=True) # 创建模型保存目录
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 初始化数据集和数据加载器
#     dataset = TumorDataSet(fla_dir="./train_data/fla", seg_dir="./train_data/seg")
#     # 检查第一个样本
#     # fla, seg = dataset[0]
#     # print("FLA 图像尺寸:", fla.shape)  # 应为 [3, 3, 240, 240]
#     # print("SEG 标签尺寸:", seg.shape)  # 应为 [3, 1, 240, 240]
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     # 初始化模型
#     model = UnetCNN(in_channels=3, out_channels=1).to(device)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     # 训练循环
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
#         # fla 输入数据 seg 标签数据
#         for batch_idx, (fla_batch, seg_batch) in enumerate(dataloader):
#             # 数据处理、模型前向传播和反向传播
#             optimizer.zero_grad() # 梯度归零
#             outputs = model(fla_batch.view(-1, *fla_batch.shape[2:]).to(device)) # 前向传播
#             loss = criterion(outputs, seg_batch.view(-1, *seg_batch.shape[2:]).to(device)) # 损失函数
#             loss.backward() # 反向传播
#             optimizer.step() # ?更新参数
#             epoch_loss += loss.item()
#             if (batch_idx + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
#         print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(dataloader):.4f}")



#     # 保存模型
#     torch.save(model,model_save_path+"/model.pth") 
#     print("模型已保存！")

# if __name__ == "__main__":
#     main()

