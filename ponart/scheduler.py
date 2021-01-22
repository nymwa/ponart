import torch.optim as optim

class WarmupScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            r = max(1e-8, step / warmup_steps)
            return min(r, r ** -0.5)
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

