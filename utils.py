class AverageMeter:
    """Tracks a running mean — used for loss logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    """Dict subclass that allows attribute-style access: d.key instead of d['key']."""

    def __getattr__(self, name):
        return self[name]
