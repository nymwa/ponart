import torch
import random as rd

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, max_tokens):
        self.dataset = dataset
        self.lengths = torch.tensor([len(sent) for sent in dataset])
        self.max_tokens = max_tokens
        self.batches = None

    def generate_batches(self):
        indices = torch.randperm(len(self.dataset))
        indices = indices[self.lengths[indices].argsort(descending=True)]
        batches = []
        batch = []
        acc = 0
        max_len = 0
        for index in indices:
            acc += 1
            this_len = self.lengths[index]
            max_len = max(max_len, this_len)
            if acc * max_len > self.max_tokens:
                batches.append(batch)
                batch = [index]
                acc = 1
                max_len = this_len
            else:
                batch.append(index)
        if batch:
            batches.append(batch)
        rd.shuffle(batches)
        return batches

    def __len__(self):
        if self.batches is None:
            self.batches = self.generate_batches()
        return len(self.batches)

    def __iter__(self):
        if self.batches is None:
            self.batches = self.generate_batches()
        for batch in self.batches:
            yield batch
        self.batches = None

