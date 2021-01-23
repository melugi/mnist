class Learner:

    def __init__(self, data, model, optimizer, loss) -> None:
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

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
    def train(self, learning_rate) -> None:
        for x, y in self.data:
            pass

    def validate(self) -> float:
        pass