import torch
from ponart.batch import Batch
from torch.nn.utils.rnn import pad_sequence as pad

class Dataset(list, torch.utils.data.Dataset):
    def __init__(self, sents, vocab):
        super().__init__(sents)
        self.vocab = vocab
        self.pad = self.vocab.pad_id
        self.cls = self.vocab.cls_id
        self.msk = self.vocab.msk_id

class MaskedLMDataset(Dataset):
    def __init__(self, sents, vocab, mask_th = 0.15, replace_th = 0.03):
        self.mask_th = mask_th
        self.replace_th = replace_th
        sents = [[vocab.cls_id] + sent for sent in sents]
        super().__init__(sents, vocab)

    def collate(self, batch):
        ei = pad([torch.tensor(sent) for sent in batch], padding_value = self.pad)
        eo = pad([torch.tensor(sent) for sent in batch], padding_value = self.pad)
        el = [len(sent) for sent in batch]
        rand_tensor = torch.rand(ei.shape)
        rand_token = torch.randint(2, len(self.vocab), ei.shape)
        normal_token = ei > 2
        position_to_mask = (rand_tensor < self.mask_th) & normal_token
        position_to_replace = (rand_tensor < self.replace_th) & normal_token
        ei.masked_fill_(position_to_mask, self.msk)
        ei.masked_scatter_(position_to_replace, rand_token)
        eo.masked_fill_(~position_to_mask, self.pad)
        return Batch(ei, eo, el)

