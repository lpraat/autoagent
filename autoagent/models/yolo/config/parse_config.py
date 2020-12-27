import yaml
import numpy as np
import copy

from autoagent.data.vision.augment import DeltaBright, HFlip


def parse_params(params_file):

    name_to_augment = {
        'delta_bright': DeltaBright(),
        'hflip': HFlip()
    }

    with open(params_file) as yaml_file:
        yaml_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    params = copy.deepcopy(yaml_params)
    params['anchor_priors'] = [np.array(x, dtype=np.float32) for x in params['anchor_priors']]
    params['augments'] = [name_to_augment[s] for s in params['augments']]

    cls_names = params['cls_names']
    params['name_to_idx'] = {k:i for k,i in zip(cls_names, range(len(cls_names)))}
    params['idx_to_name'] = {v:k for k,v in params['name_to_idx'].items()}
    params['num_cls'] = len(cls_names)

    return yaml_params, params
