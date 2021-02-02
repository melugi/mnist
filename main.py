import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import Tensor

from SgdOptimizer import SgdOptimizer
from Learner import Learner
import utils

def initalize_params(size, std=1.0) -> Tensor: return (torch.rand(size)*std).requires_grad_()

def mse(predictions, targets) -> Tensor: return ((predictions - targets)**2).mean().sqrt()

def mnist_loss(predictions, targets) -> Tensor:
        predictions = torch.sigmoid(predictions)
        return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb) -> float:
        predictions = xb.sigmoid()
        correct = (predictions > 0.5) == yb
        return correct.float().mean()

val_set = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True, 
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,))
                                    ])),
    batch_size=64, shuffle=True)

train_set = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True, 
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,))
                                    ])),
    batch_size=500, shuffle=True)

samples, labels = next(iter(train_set))
print(samples.shape)
print(labels.shape)

utils.peek_into_dataloader(train_set)

'''
linear_model = nn.Linear(28*28, 1)
optimizer = SgdOptimizer(linear_model.parameters(), 1)

learner = Learner(train_set, linear_model, optimizer, mnist_loss, batch_accuracy)
learner.fit(10)

'''