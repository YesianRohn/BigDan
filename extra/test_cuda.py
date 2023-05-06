import torch
import os

#cuda是否可用
print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
# 返回gpu数量
print(f'torch.cuda.device_count:{torch.cuda.device_count()}')
# 返回gpu名字，设备索引默认从0开始
print(f'torch.cuda.get_device_name:{torch.cuda.get_device_name(0)}')
