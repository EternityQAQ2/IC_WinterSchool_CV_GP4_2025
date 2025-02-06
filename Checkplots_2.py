# 在训练集的两个nii文件中，0d3_fla.nii代表的是大脑三视图+3d图。 0d3_seg.nii代表的是大脑的分割图,癌变地区的图。
# 一个nlb文件包含 image数据、一个仿射数组 affine、image元数据header
import nibabel as nb
import os
import numpy as np
import matplotlib.pyplot as plt #图像处理
import cv2

# 在 nibabel 读取 .nii 文件时，默认数据类型通常是 float64，
# 而肿瘤区域的标注数据 (seg.nii) 可能是 0（背景）和 1（肿瘤）。
# 但是 cv2.imwrite() 需要的数据范围是 0-255 的整数 (uint8)，
# 如果数据是 float64 的 0 和 1，cv2.imwrite() 可能会把 1 也当作 0，
# 从而导致全黑图片。


def save_slices(slices, paths):
    for img_slice, path in zip(slices, paths):
        # img_slice = normalize_to_uint8(img_slice)  # 归一化并转换
        img_slice = contrast_stretching(img_slice)  # 拉伸对比度
        cv2.imwrite(path, img_slice)

def contrast_stretching(img_slice):
    # 将图像拉伸到 0-255 范围
    stretched_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX)
    return stretched_slice.astype(np.uint8)

# 归一化 I-Imin / Imax-Imin * 255 ，
def normalize_to_uint8(img_slice):
    # 找到最小值和最大值
    min_val = np.min(img_slice)
    max_val = np.max(img_slice)
    # 防止除以 0 的问题
    if max_val - min_val == 0:
        return np.zeros_like(img_slice, dtype=np.uint8)
    # 归一化到 0-1，然后映射到 0-255
    norm_slice = (img_slice - min_val) / (max_val - min_val)
    return (norm_slice * 255).astype(np.uint8)

def show_slices(slices):
   """ 显示一行图像切片 """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")       

def show_pictures(img_fla_data, img_seg_data, i):
    img_flat_slice_0 = np.fliplr(np.rot90(img_fla_data[120, :, :],1))  # 冠状面（YZ） 固定x=120 yz任意，因为图片240，240，155。顺时针旋转90度反转
    img_flat_slice_1 = np.fliplr(np.rot90(img_fla_data[:, 120, :],1))  # 矢状面（XZ） 顺时针旋转90度并左右翻转
    img_flat_slice_2 = np.rot90(img_fla_data[:, :, 78],3)   # 轴状面（XY） 顺时针旋转270度

    # 创建保存路径
    fla_path = f"./fla_slices/{i}"
    os.makedirs(fla_path, exist_ok=True)  # 确保目录存在

    # 定义保存的文件名
    fla_save_paths = [
        f"{fla_path}/fla_{i}_(YZ).png",
        f"{fla_path}/fla_{i}_(XZ).png",
        f"{fla_path}/fla_{i}_(XY).png"
    ]
    save_slices([img_flat_slice_0, img_flat_slice_1, img_flat_slice_2], fla_save_paths)
    # 显示图片（可选）
    

    # 获得三个维度的切片(seg)
    img_seg_slice_0 = np.fliplr(np.rot90(img_seg_data[120, :, :],1)) #顺时针旋转90度
    img_seg_slice_1 = np.fliplr(np.rot90(img_seg_data[:, 120, :],1)) #顺时针旋转90度并左右翻转
    img_seg_slice_2 = np.rot90(img_seg_data[:, :, 78],3) #顺时针旋转270度

    seg_path = f"./seg_slices/{i}"
    os.makedirs(seg_path, exist_ok=True)  # 确保目录存在
    seg_save_paths = [
        f"{seg_path}/seg_{i}_(YZ).png",
        f"{seg_path}/seg_{i}_(XZ).png",
        f"{seg_path}/seg_{i}_(XY).png"
    ]
    # show_slices([img_seg_slice_0, img_seg_slice_1, img_seg_slice_2])
    # plt.show()
    save_slices([img_seg_slice_0, img_seg_slice_1, img_seg_slice_2], seg_save_paths)


    
path = "./dataset_segmentation/train" # 训练集路径
for i in range(1,211):
    folder_name = f"{i:03d}" # 生成 '001', '002', ..., '210'
    folder_path =os.path.join(path,folder_name)
    img_fla_path = os.path.join(folder_path,folder_name+"_fla.nii") # 大脑三视图+3d图路径
    img_seg_path = os.path.join(folder_path,folder_name+"_seg.nii") # 大脑的分割图路径
    
    
    if img_fla_path is None:
        print(f"文件不存在: {img_fla_path}")
    if img_seg_path is None:
        print(f"文件不存在: {img_seg_path}")

    img_fla = nb.load(img_fla_path) # 打开fla文件，即大脑三视图+3d图
    img_seg = nb.load(img_seg_path) # 打开seg文件，即大脑的分割图

    img_fla_data = img_fla.get_fdata() # 获取fla文件ndarray类型数据
    img_seg_data = img_seg.get_fdata() # 获取seg文件ndarray类型数据 

    # 三维数据， x,y,z
    # (240,240,155)
    
    # 获得三个维度的切片(flat)和(seg)
    show_pictures(img_fla_data,img_seg_data,i)



    


