import torch
from numpy import *
import torchvision

test_set = torchvision.datasets.MNIST('./', train=False, download=True, transform=None)
train_set = torchvision.datasets.MNIST('./', train=True, download=True, transform=None)
  