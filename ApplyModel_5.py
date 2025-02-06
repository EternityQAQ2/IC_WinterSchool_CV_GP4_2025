# 5 使用已经训练好的模型进行预测
import os
import torch
from TrainModels_4 import UnetCNN
import cv2
import numpy as np

TARGET_SIZE = (240, 240)   
model_save_path = "./model"
Results_path = "./Results"
input_root = "./fla_slices" # 输入数据路径



model = torch.load(os.path.join(model_save_path, "model.pth"), weights_only=False) #模型可信！
if model is not None:
    print("模型加载成功")


model.eval()  # 切换到评估模式,使得模型BN层等失效

# 预处理图像
def preprocess_image(image,TARGET_SIZE):
    image = cv2.resize(image, TARGET_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换成RGB 通道
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)) # 转换成CxHxW
    tensor = torch.from_numpy(image)
    return tensor

for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder) 
    if not os.path.isdir(folder_path):
        continue

    fla_images = []