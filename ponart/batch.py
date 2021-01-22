class Batch:
    def __init__(self, ei, eo = None, el = None):
        self.encoder_inputs = ei
        self.encoder_outputs = eo
        self.encoder_lengths = el

    def __len__(self):
        return self.encoder_inputs.shape[1]

    def cuda(self):
        self.encoder_inputs = self.encoder_inputs.cuda()
        if self.encoder_outputs is not None:
            self.encoder_outputs = self.encoder_outputs.cuda()
        return self

