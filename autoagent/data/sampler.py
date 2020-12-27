import torch
import numpy as np


class MultiScaleBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for multi-scale training.
    """
    def __init__(self, sampler, batch_size, drop_last, scales, multiscale_every=1):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.scales = scales
        self.multiscale_every = multiscale_every

    def __len__(self):
        sampler_len = len(self.sampler)
        l = sampler_len // self.batch_size
        return l if sampler_len % self.batch_size == 0 or self.drop_last else l+1

    def __iter__(self):
        n = 0
        img_size = np.random.choice(self.scales)
        batch = []

        sampler_len = len(self.sampler)
        remaining = sampler_len
        sampler_iter = iter(self.sampler)
        while self.batch_size < remaining:
            batch = [(next(sampler_iter), img_size) for _ in range(self.batch_size)]
            yield batch
            n += 1
            remaining = sampler_len - n * self.batch_size

            if n % self.multiscale_every == 0:
                img_size = np.random.choice(self.scales)


        if remaining > 0 and not self.drop_last:
            yield [(i, img_size) for i in sampler_iter]
