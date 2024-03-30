import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 获取当前活动的CUDA设备索引
current_device = torch.cuda.current_device()

# 获取当前活动设备的名称
current_device_name = torch.cuda.get_device_name(current_device)

print(f"Current CUDA Device Index: {current_device}")
print(f"Current CUDA Device Name: {current_device_name}")


print(os.environ.get("CUDA_HOME"))
print(torch.__version__)
