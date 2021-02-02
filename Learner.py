import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader

class Learner:

    def __init__(self,
                data: DataLoader,
                model,
                optimizer,
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
        for (x, y) in iter(self.data):
            self.calc_grad(x.view(-1, 28*28), y)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def calc_grad(self, x: Tensor, y: Tensor) -> None:
        prediction = self.model(x)
        loss = self.loss_func(prediction, y)
        loss.backward()

    def validate(self) -> float:
        accs = []
        for (xb, yb) in iter(self.data):
            preds = self.model(xb.view(-1, 28*28))
            print(preds.size(), preds.shape, preds)
            accs += self.metric_func(preds, yb)
        
        return round(torch.stack(accs).mean().item(), 4)

