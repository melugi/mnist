import torch
from torch.functional import Tensor
from numpy import *
import torchvision

def initalize_params(size, std=1.0) -> Tensor: return (torch.rand(size)*std).requires_grad_()

def mse(predictions, targets) -> Tensor: return ((predictions - targets)**2).mean().sqrt()

def mnist_loss(predictions, targets) -> Tensor:
        predictions = torch.sigmoid(predictions)
        return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb) -> float:
        predictions = xb.sigmoid()
        correct = (predictions > 0.5) == yb
        return correct.float().mean()

test_set = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True, 
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
    batch_size=64, shuffle=True)

train_set = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True, 
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
    batch_size=500, shuffle=True)

weights = initalize_params(28*28)
bias = initalize_params(1)
