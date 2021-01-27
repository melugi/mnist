import torch
from torch.functional import Tensor

class Learner:

    def __init__(self, data, model, optimizer, loss_func, metric_func) -> None:
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metric_func = metric_func

    def fit(self, epochs, learning_rate) -> None:
        for i in range(epochs):
            self.train(learning_rate)
            print(self.validate(), end=' ')

    '''
    TODO: Write train function to:
        - Loop through data (Data contains X, Y tuples. X is the prediction, Y is the true result)
        - Make predictions
        - Calculate the loss
        - Calculate gradient
        - Use the gradient to step down on each parameter
    '''
    def train(self) -> None:
        for x, y in self.data:
            self.calc_grad(x, y)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def calc_grad(self, x, y):
        prediction = self.model(x)
        loss = self.loss_func(prediction, y)
        loss.backward()

    def batch_accuracy(self, xb, yb) -> float:
        pass

    def validate(self) -> float:
        pass

