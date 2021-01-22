class Accumulator:
    def __init__(self):
        self.num_examples = 0
        self.loss = 0.0

    def update(self, batch_size, loss):
        self.num_examples += batch_size
        self.loss += loss * batch_size

    def total(self):
        return self.loss / self.num_examples

