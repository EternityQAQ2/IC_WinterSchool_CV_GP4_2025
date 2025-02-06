# # 5 使用已经训练好的模型进行预测
# import os
# import torch
# from TrainModels_4 import UnetCNN
# import cv2
# import numpy as np

# TARGET_SIZE = (240, 240)   
# model_save_path = "./model"
# Results_path = "./Results"
# input_root = "./fla_slices" # 输入数据路径



# # 预处理图像
# def preprocess_image(image,TARGET_SIZE):
#     image = cv2.resize(image, TARGET_SIZE)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换成RGB 通道
#     image = image.astype(np.float32) / 255.0
#     image = np.transpose(image, (2, 0, 1)) # 转换成CxHxW
#     tensor = torch.from_numpy(image).unsqueeze(0)  # 添加 batch 维度
#     return tensor

# def postprocess_mask(prob_map, threshold=0.5):
#     """
#     prob_map: 模型输出的 numpy 数组，形状 (H, W)
#     返回：二值 mask 和提取到的不规则轮廓
#     """
#     # 阈值化
#     mask = (prob_map >= threshold).astype(np.uint8) * 255  # 0或255
#     # 使用 OpenCV 查找轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return mask, contours

# model = torch.load(os.path.join(model_save_path, "model.pth"), weights_only=False) #模型可信！
# if model is not None:
#     print("模型加载成功")

# model.eval()  # 切换到评估模式,使得模型BN层等失效

# for folder in sorted(os.listdir(input_root)):

#     Result_save_paths = [
#         f"{Results_path}/fla_{folder}_(YZ).png",
#         f"{Results_path}/fla_{folder}_(XZ).png",
#         f"{Results_path}/fla_{folder}_(XY).png"
#         ]
    

#     folder_path = os.path.join(input_root, folder)  # ./fla_slices/1
#     if not os.path.isdir(folder_path):
#         continue
#     image_files = os.listdir(folder_path) # ['fla_1_(XY).png', 'fla_1_(XZ).png', 'fla_1_(YZ).png']
#     fla_images = []

#     # 获取三张
#     for images in image_files:
#         image_path = os.path.join(folder_path, images)
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR读取
#         if image is None:
#             print(f"文件不存在: {image_path}")
#         image = preprocess_image(image, TARGET_SIZE) # BGR-RGB 转换
#         fla_images.append(image)
#     fla_images_tensor = torch.cat(fla_images, dim=0)

#     with torch.no_grad():
#         output = model(fla_images_tensor)  # 输出形状为 (1, out_channels, H, W)

#     prob_map = output[0, 0].cpu().numpy() # 获得概率图
#     mask, contours = postprocess_mask(prob_map, threshold=0.5)

#     for images in image_files:
#         i = 0
#         orig_img_path = os.path.join(folder_path, images)
#         orig_img = cv2.imread(orig_img_path)
#         orig_img = cv2.resize(orig_img, (mask.shape[1], mask.shape[0]))
#         # 绘制轮廓到原图上
#         overlay = orig_img.copy()
#         cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # 红色轮廓

#         #叠加 ？    
#         mask_color = np.zeros_like(orig_img)
#         mask_color[:, :] = (0, 255, 0)  # 绿色
#         alpha = 0.3
#         overlay[mask > 0] = cv2.addWeighted(overlay, 1, mask_color, alpha, 0)[mask > 0]
        
#         out_folder = os.path.join(Results_path, folder) # ./Results/1
#         os.makedirs(out_folder, exist_ok=True)
#         out_img_path = os.path.join(out_folder,Result_save_paths[i])# ./Results/1/fla_1_(YZ).png
#         i += 1
#         cv2.imwrite(out_img_path, overlay)
#         print(f"保存结果到 {out_img_path}")

# print("预测完成！")
import os
import torch
from TrainModels_4 import UnetCNN
import cv2
import matplotlib.pyplot as plt
import numpy as np

TARGET_SIZE = (240, 240)
model_save_path = "./model"
Results_path = "./Results"
input_root = "./fla_slices"

def preprocess_image(image, TARGET_SIZE):
    image = cv2.resize(image, TARGET_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor

def postprocess_mask(prob_map, threshold=0.5):
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

model = torch.load(os.path.join(model_save_path, "model.pth"),weights_only=False)
model.eval()

for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    fla_images = []
    valid_files = []

    for img_file in image_files:
        image_path = os.path.join(folder_path, img_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        tensor = preprocess_image(image, TARGET_SIZE)
        fla_images.append(tensor)
        valid_files.append(img_file)
    
    if not fla_images:
        continue

    fla_images_tensor = torch.cat(fla_images, dim=0)
    with torch.no_grad():
        output = torch.sigmoid(model(fla_images_tensor))  # 应用Sigmoid

    for idx in range(len(valid_files)):
        img_file = valid_files[idx]
        prob_map = output[idx, 0].cpu().numpy()

        plt.imshow(prob_map, cmap="gray")
        plt.title("Probability Map")
        plt.show()
        
        mask, contours = postprocess_mask(prob_map)

        orig_img = cv2.imread(os.path.join(folder_path, img_file))
        orig_img = cv2.resize(orig_img, (mask.shape[1], mask.shape[0]))
        overlay = orig_img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # 创建绿色半透明叠加层
        mask_color = np.zeros_like(overlay, dtype=np.uint8)
        mask_color[mask == 255] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1, mask_color, 0.3, 0)

        # 确定保存路径
        save_dir = os.path.join(Results_path, folder)
        os.makedirs(save_dir, exist_ok=True)
        if "(YZ)" in img_file:
            save_path = os.path.join(save_dir, f"fla_{folder}_(YZ).png")
        elif "(XZ)" in img_file:
            save_path = os.path.join(save_dir, f"fla_{folder}_(XZ).png")
        elif "(XY)" in img_file:
            save_path = os.path.join(save_dir, f"fla_{folder}_(XY).png")
        else:
            print(f"未知的切片类型: {img_file}")
            continue

        cv2.imwrite(save_path, overlay)
        print(f"结果已保存至: {save_path}")

print("预测完成！")