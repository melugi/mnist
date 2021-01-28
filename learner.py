import torch
from optimizer import Optimizer
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader

class Learner:

    def __init__(self,
                data: DataLoader,
                model,
                optimizer: Optimizer,
                loss_func,
                metric_func
                ) -> None:
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metric_func = metric_func

    def fit(self, epochs: int) -> None:
        for i in range(epochs):
            self.train()
            print(self.validate(), end=' ')

    def train(self) -> None:
        for x, y in self.data:
            self.calc_grad(x, y)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def calc_grad(self, x: Tensor, y: Tensor) -> None:
        prediction = self.model(x)
        loss = self.loss_func(prediction, y)
        loss.backward()

    def batch_accuracy(self, xb, yb) -> float:
        predictions = xb.sigmoid()
        correct = (predictions > 0.5) == yb
        return correct.float().mean()

    def validate(self) -> float:
        accs = [self.batch_accuracy(xb, yb) for xb, yb in self.data]
        return round(torch.stack(accs).mean().item(), 4)

