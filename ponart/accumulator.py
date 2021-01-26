class Accumulator:
    def __init__(self, epoch, num_batches):
        self.epoch = epoch
        self.num_batches = num_batches
        self.num_list = []
        self.loss_list = []
        self.wpb_list = []
        self.spb_list = []
        self.lr_list = []
        self.grad_list = []

    def update(self, batch, loss, lr, grad):
        self.num_list.append(len(batch))
        self.loss_list.append(loss.item())
        self.wpb_list.append(sum(batch.encoder_lengths))
        self.spb_list.append(len(batch.encoder_lengths))
        self.lr_list.append(lr)
        self.grad_list.append(grad)

    def step_log(self):
        return '| inner | epoch {}, {}/{} | loss {:.4f} | lr {:.4e} | grad {:.4f} | w/b {} | s/b {}'.format(
                self.epoch,
                len(self.num_list),
                self.num_batches,
                self.loss_list[-1],
                self.lr_list[-1],
                self.grad_list[-1],
                self.wpb_list[-1],
                self.spb_list[-1],
                )

    def avg(self, lst):
        num_examples = sum(self.num_list)
        return sum([n * x for n, x in zip(self.num_list, lst)]) / num_examples

    def epoch_log(self, num_steps):
        return '| train | epoch {} | loss {:.4f} | lr {:.4e} | grad {:.4f} | w/b {:.1f} | s/b {:.1f} | steps {}'.format(
                self.epoch,
                self.avg(self.loss_list),
                self.avg(self.lr_list),
                self.avg(self.grad_list),
                self.avg(self.wpb_list),
                self.avg(self.spb_list),
                num_steps)

