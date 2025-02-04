import os
import gzip
import shutil

# 定义数据集的主目录
base_dir = "./dataset_segmentation/train"  # 修改为你的数据路径

# 遍历 001 到 210 号文件夹
for i in range(1, 211):
    folder_name = f"{i:03d}"  # 生成 '001', '002', ..., '210'
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".nii.gz"):
                gz_file_path = os.path.join(folder_path, file_name)
                nii_file_path = os.path.join(folder_path, file_name.replace(".gz", ""))

                # 解压缩 .nii.gz 文件
                with gzip.open(gz_file_path, "rb") as f_in:
                    with open(nii_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"解压完成: {gz_file_path} -> {nii_file_path}")

    else:
        print(f"文件夹不存在: {folder_path}")
