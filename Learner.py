import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader

class Learner:

    def __init__(self,
                training_data: DataLoader,
                val_data: DataLoader,
                model,
                optimizer,
                loss_func,
                metric_func
                ) -> None:
        self.training_data = training_data
        self.val_data = val_data
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metric_func = metric_func

    def fit(self, epochs: int) -> None:
        accs = []
        for i in range(epochs):
            print("===== Epoch {} =====".format(i + 1))
            self.train()
            accs.append(self.validate())

        [print("== Epoch {} Accuracy: {}%".format(idx+1, acc)) for idx, acc in enumerate(accs)]
        print("==== Last 10 Runs Accuracy: {}% ====".format(sum(accs[-10:])/10))
        print("==== Total Accuracy: {}% ====".format(sum(accs)/len(accs)))

    def train(self) -> None:
        running_loss = 0
        for (images, labels) in self.training_data:
            images = images.view(-1, 28*28)
            running_loss += self.calc_grad(images, labels)
            self.optimizer.step()
            self.optimizer.zero_grad()

        average_loss = running_loss/len(self.training_data)
        print("Training loss: {}".format(round(average_loss, 8)))

    def calc_grad(self, images: Tensor, labels: Tensor) -> float:
        predictions = self.model(images)
        loss = self.loss_func(predictions, labels)
        loss.backward()
        return loss.item()

    def validate(self) -> float:
        acc = 0
        for (images, labels) in self.val_data:
            images = images.view(-1, 28*28)
            with torch.no_grad():
                preds = self.model(images)

            torch.exp_(preds)
            acc = self.metric_func(preds, labels)

        acc = round(acc, 4)*100
        print("Current Accuracy: {}%".format(acc))
        return acc

