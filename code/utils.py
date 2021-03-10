import torch
from numpy import random

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch is using {Device.type.upper()}\n')


def fix_random(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
