import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ponart.util import make_tokenizer
from ponart.dataset import MaskedLMDataset
from ponart.sampler import Sampler
from ponart.ponart import Ponart
from ponart.scheduler import WarmupScheduler
from ponart.accumulator import Accumulator
from ponart.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def make_dataset(tokenizer):
    with open('data/tatoeba.pickle', 'rb') as f:
        data = pickle.load(f)
    dataset = MaskedLMDataset(data, tokenizer.vocab)
    return dataset

def main():
    tokenizer = make_tokenizer()
    dataset = make_dataset(tokenizer)
    sampler = Sampler(dataset, 30000)
    loader = DataLoader(dataset, batch_sampler = sampler, collate_fn = dataset.collate)
    model = Ponart(len(tokenizer.vocab))
    model = model.cuda()
    print('#params (to train): {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('#params (total): {}'.format(sum(p.numel() for p in model.parameters())))
    optimizer = optim.AdamW(model.parameters(), lr = 0.002, weight_decay = 0.01)
    scheduler = WarmupScheduler(optimizer, 8000)
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.pad)

    with open('train.log', 'w') as f:
        f.write('')

    clip_norm = 1.0
    num_steps = 0
    model.train()
    for epoch in range(3000):
        accum = Accumulator(epoch, len(loader))
        for step, batch in enumerate(loader):
            batch.cuda()
            pred = model(batch.encoder_inputs)
            pred = pred.view(-1, pred.size(-1))
            loss = criterion(pred, batch.encoder_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            num_steps += 1
            lr = scheduler.get_last_lr()[0]
            accum.update(batch, loss, lr, grad)
            logger.info(accum.step_log())
        logger.info(accum.epoch_log(num_steps))
    torch.save(model.state_dict(), 'checkpoints/pretrain.pt')

if __name__ == '__main__':
    main()

