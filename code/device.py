import torch

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch is using {Device.type.upper()}\n')
