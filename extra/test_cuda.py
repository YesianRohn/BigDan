import torch
import os

print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
os.system('nvidia-smi')
