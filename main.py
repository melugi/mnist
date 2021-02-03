import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.functional import Tensor

from SgdOptimizer import SgdOptimizer
from Learner import Learner
import utils

def initalize_params(size, std=1.0) -> Tensor: return (torch.rand(size)*std).requires_grad_()

def mse(predictions, targets) -> Tensor: return ((predictions - targets)**2).mean().sqrt()

def mnist_loss(predictions, targets) -> Tensor:
        predictions = torch.sigmoid(predictions)
        return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(batch_images, batch_labels) -> float:
        correct = 0
        for idx, image in enumerate(batch_images):
            prediction = torch.argmax(image).item()
            label = batch_labels[idx].item()
            if prediction == label:
                correct += 1
        
        return correct/len(batch_images)

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

model = nn.Sequential(nn.Linear(28*28, 2**7),
                      nn.ReLU(),
                      nn.Linear(2**7, 2**6),
                      nn.ReLU(),
                      nn.Linear(2**6, 10),
                      nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

learner = Learner(train_set, val_set, model, optimizer, nn.NLLLoss(), batch_accuracy)
learner.fit(10)
