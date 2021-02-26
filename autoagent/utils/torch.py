import numpy as np
import torch

from functools import reduce


def get_num_params(model):
    """
    Get the total number trainable parameters in model.
    """
    num_params = 0
    for param in model.parameters():
        num_params += reduce(lambda x, y: x*y, param.shape, 1)
    return num_params


def warmup_params(curr_iter, optim, keys, xp, fps):
    """
    Warmup params (keys) in optimizer using linear interpolation
    on (xp, fp) data points.
    """
    for param_group in optim.param_groups:
        for i, k in enumerate(keys):
            param_group[k] = np.interp(curr_iter, xp, fps[i])


def assign_flat_params(model, flat_params):
    """
    Assigns the flattened parameter vector to the model.
    Parameters
    ----------
    model : torch.nn.Module
        Parameterized model to update.
    flat_params : torch.Tensor
        Flattened parameter vector.
    """
    sections = [torch.reshape(w, (-1,)).shape[0] for w in model.parameters()]
    params = torch.split(flat_params, sections, dim=0)

    for i, p in enumerate(model.parameters()):
        p.data = torch.reshape(params[i], p.shape)