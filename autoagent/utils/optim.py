import torch
import math
import copy


class EMA:
    """
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py
    """

    def __init__(self, model, updates, decay=0.9999, exp_d=2000, mode='exp'):
        self.ema = copy.deepcopy(model).eval()

        if mode == 'exp':
            self.decay_f = lambda x: decay * (1 - math.exp(-x / exp_d))
        else:
            self.decay_f = lambda x: min(decay, (1+decay)/(10+decay))

        self.updates = updates

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay_f(self.updates)

        model_dict = model.state_dict()
        for k, v in self.ema.state_dict().items():
            # Iterate over dict to get not only params but also buffers
            if v.dtype.is_floating_point:
                v.mul_(decay)
                v.add_((1-decay) * model_dict[k])
