import torch

# def check_torch_gpu():
#     print("PyTorch 版本:", torch.__version__)
#     print("PyTorch 是否支持 CUDA:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         device_count = torch.cuda.device_count()
#         print("可用 GPU 数量:", device_count)
#         for i in range(device_count):
#             print(f"GPU 设备 {i} 名称:", torch.cuda.get_device_name(i))
#     else:
#         print("未检测到可用的 GPU，当前仅可使用 CPU 进行计算。")

# if __name__ == "__main__":
#     check_torch_gpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
