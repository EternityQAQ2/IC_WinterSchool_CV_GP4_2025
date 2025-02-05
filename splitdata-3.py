# 分验证集和训练集 80% 20%

import os
import shutil
import random

# 复制数据
def copy_folders(source_dir,folder_list, target_dir):
    for folder in folder_list:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(target_dir, folder)
        shutil.copytree(src, dst)  

Fla_source_path = "./fla_slices"
Seg_source_path = "./seg_slices"  
Train_save_path = ["./train_data/fla", "./train_data/seg"]
Val_save_path = ["./val_data/fla", "./val_data/seg"]

# 创建文件夹
for path in Train_save_path:
    # 如果文件夹存在，删除文件夹
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
for path in Val_save_path:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# 遍历文件夹
# 获取所有子文件夹（每个文件夹是一组数据）
folders_Fla = sorted(os.listdir(Fla_source_path)) # 返回的是次级目录的名
# print(folders_Fla)

random.seed(42) # 随机打乱
random.shuffle(folders_Fla)
# print(folders_Fla)

split_Index = int(0.8 * len(folders_Fla)) # 80% 20% 分割
train_folders_Fla = folders_Fla[:split_Index] #从0到split_Index 训练集
val_folders_Fla = folders_Fla[split_Index:] #从split_Index到最后 验证集
'''
[1,2,5,6,7,8,9,10] -- 训练集文件名
[3,4] -- 测试集文件名
'''
copy_folders(Fla_source_path, train_folders_Fla, Train_save_path[0])
copy_folders(Fla_source_path, val_folders_Fla, Val_save_path[0])
copy_folders(Seg_source_path, train_folders_Fla, Train_save_path[1])
copy_folders(Seg_source_path, val_folders_Fla, Val_save_path[1])