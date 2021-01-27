import torch
from torch.functional import Tensor
from numpy import *
import torchvision

def initalize_params(size, std=1.0) -> Tensor: return (torch.rand(size)*std).requires_grad()

def mse(predictions, targets) -> Tensor: return ((predictions - targets)**2).mean().sqrt()

def mnist_loss(self, predictions, targets) -> Tensor:
        predictions = torch.sigmoid(predictions)
        return torch.where(targets==1, 1-predictions, predictions).mean()

test_set = torchvision.datasets.MNIST('./', train=False, download=True, transform=None)
train_set = torchvision.datasets.MNIST('./', train=True, download=True, transform=None)

