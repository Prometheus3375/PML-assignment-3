import torch

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
