import torch
from numpy import random

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch is using {Device.type.upper()}\n')


def make_determenistic(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
