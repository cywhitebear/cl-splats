import torchtyping
import torch

Image = torchtyping.TensorType["H", "W", 3, torch.float32]