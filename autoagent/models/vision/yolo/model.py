import torch
import torch.nn as nn

from autoagent.models.vision.yolo.layers import Conv
from autoagent.utils.vision import compute_generalized_iou
from autoagent.models.vision.yolo.utils import compute_final_bboxes


class Yolo():

    def __init__(self, params):
        self.params = params

        s_to_mults = {
            'small': (0.5, 0.33),
            'medium': (0.75, 0.66),
            'large': (1.0, 1.0)
        }
        version, size = self.params['version'].split("-")
        if version == 'v5':
            from autoagent.models.vision.yolo.model_v5 import YoloModel
            mult_c, mult_d = s_to_mults[size]
        else:
            raise NotImplementedError("Yolo model not found.")

        s_to_act = {
            'leaky': nn.LeakyReLU,
            'hswish': nn.Hardswish
        }

        s_to_conv = {
            'conv': Conv
        }

        self.model = YoloModel(
            num_classes=self.params['num_cls'],
            mult_c=mult_c,
            mult_d=mult_d,
            activation=s_to_act[self.params['act']],
            conv=s_to_conv[self.params['conv']]
        )

        if self.params['init_biases']:
            self.model.head.init_biases()

        self.det_cache = None

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cuda(self):
        return self.model.cuda()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def half(self):
        return self.model.half()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, checkpoint, fine_tune=False):
        n1 = ('head.fs', 'head.fm', 'head.fl')
        n2 = ('weight', 'bias')
        to_remove = [f"{x}.{y}" for x in n1 for y in n2] if fine_tune else []
        self_state_dict = self.state_dict()
        if list(checkpoint.keys())[0] in self_state_dict:
            for key in to_remove:
                checkpoint[key] = self_state_dict[key]
            self.model.load_state_dict(checkpoint)
        else:
            # Import weights from yolov5 repo
            self._load_original_v5_weights(checkpoint, to_remove)

    def _load_original_v5_weights(self, v5_state_dict, to_remove):
        self_state_dict = self.state_dict()
        keys = list(self_state_dict.keys())

        corrected_state_dict = {}
        idx = 0
        for (k, v) in v5_state_dict.items():
            if 'anchor' not in k:
                assert k.split(".")[-1] == keys[idx].split(".")[-1]
                corrected_state_dict[keys[idx]] = v
                idx += 1

        for key in to_remove:
            corrected_state_dict[key] = self_state_dict[key]

        self.model.load_state_dict(corrected_state_dict)

        for m in self.model.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU]:
                m.inplace = True

    @torch.no_grad()
    def detect(self, x, conf_thresh, use_cache=False, with_ids=False):
        # Build prediction grid
        num_anchors = len(self.params['anchor_priors'])
        img_dims = (x.shape[2], x.shape[3])

        if self.det_cache is None:
            aux_grids = []
            steps = []
            anchor_priors = []
        else:
            aux_grids, steps, anchor_priors = self.det_cache

        preds = self(x)
        det_grids = []
        for i in range(3):
            gy, gx = preds[i].shape[2:4]
            pred_grid = preds[i].view(
                preds[i].shape[0], num_anchors, 5+self.params['num_cls'], gy, gx
            ).permute(0, 1, 3, 4, 2).contiguous()
            det_grids.append(pred_grid.clone())

            if self.det_cache is None:
                # Auxiliary grids
                grid_y, grid_x = torch.meshgrid([torch.arange(gy), torch.arange(gx)])
                aux_grids.append(torch.stack([grid_x, grid_y], dim=2).view((1, 1, gy, gx, 2)).to(preds[i].device))

                # Steps
                steps.append(torch.tensor([img_dims[0] / gy, img_dims[1] / gx], dtype=torch.float32).to(preds[i].device))

                # Anchor priors
                anchor_priors.append(torch.from_numpy(self.params['anchor_priors'][2-i]).view(1, num_anchors, 1, 1, 2).to(preds[i].device))

        if use_cache and self.det_cache is None:
            self.det_cache = (aux_grids, steps, anchor_priors)

        # Retrieve bboxes from prediction grid
        return compute_final_bboxes(det_grids, conf_thresh, anchor_priors, aux_grids, steps,
                                    with_ids=with_ids, bbox_fn=self.params['bbox_fn'])


    def get_loss(self, preds, targets):
        # Assumption: targets and preds elements are on same device
        device = preds[0].device
        loss_loc = torch.zeros(1, device=device)
        loss_det = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        reduction = self.params['reduction']
        bcent_det = nn.BCEWithLogitsLoss(reduction=reduction).to(device)
        bcent_cls = nn.BCEWithLogitsLoss(reduction=reduction).to(device)

        batch_size = preds[0].shape[0]

        for i in range(3):
            pred = preds[i]

            target, img_idx, x_idx, y_idx, anchor_idx = targets[i]
            target = target.permute(0, 3, 1, 2, 4)

            gy, gx = target.shape[2:4]
            pred_grid = pred.view(
                batch_size, 3, 5+self.params['num_cls'], gy, gx
            ).permute(0, 1, 3, 4, 2).contiguous()

            pred = pred_grid[img_idx, anchor_idx, y_idx, x_idx]

            # Target grid and target values
            target = target[img_idx, anchor_idx, y_idx, x_idx]
            target_grid = torch.zeros_like(pred_grid, device=device)

            if pred.shape[0] > 0:
                # x, y, w, h
                pred_x = 2*torch.sigmoid(pred[:, 0]) - 0.5
                target_x = target[:, 0]
                pred_y = 2*torch.sigmoid(pred[:, 1]) - 0.5
                target_y = target[:, 1]
                pred_w = pred[:, 2]
                target_w = target[:, 2]
                pred_h = pred[:, 3]
                target_h = target[:, 3]

                # Localization
                anchors = torch.tensor(self.params['anchor_priors'][2-i], device=device)
                scaled_anchors = anchors / self.params['steps'][i]

                if self.params['bbox_fn'] == 'exp':
                    pred_bboxes = torch.stack([
                        pred_x, pred_y,
                        torch.exp(pred_w).clamp(max=1e3) * scaled_anchors[anchor_idx, 0],
                        torch.exp(pred_h).clamp(max=1e3) * scaled_anchors[anchor_idx, 1]
                    ]).t()
                else:
                    pred_bboxes = torch.stack([
                        pred_x, pred_y,
                        (2*torch.sigmoid(pred_w))**2 * scaled_anchors[anchor_idx, 0],
                        (2*torch.sigmoid(pred_h))**2 * scaled_anchors[anchor_idx, 1]
                    ]).t()

                target_bboxes = torch.stack([
                    target_x, target_y, target_w, target_h
                ]).t()

                cious = compute_generalized_iou(pred_bboxes, target_bboxes, format='xywh', kind='ciou')
                if reduction == 'sum':
                    loss_loc += torch.sum(1 - cious) / batch_size
                else:
                    loss_loc += torch.mean(1 - cious)

                # Detection
                ciou_ratio = self.params['ciou_ratio']
                cious = cious.detach().clamp(min=0).type(target_grid.dtype)
                target_grid[img_idx, anchor_idx, y_idx, x_idx, 4] = (1 - ciou_ratio) + ciou_ratio * cious

                # Classification
                if self.params['num_cls'] > 1:
                    pred_cls = pred[:, 5:]
                    target_cls = target[:, 5:]
                    if reduction == 'sum':
                        loss_cls += bcent_cls(pred_cls, target_cls) / batch_size
                    else:
                        loss_cls += bcent_cls(pred_cls, target_cls)

            # Detection
            pred_conf = pred_grid[..., 4]
            target_conf = target_grid[..., 4]
            if reduction == 'sum':
                loss_det += bcent_det(pred_conf, target_conf) * self.params['balance'][i] / batch_size
            else:
                loss_det += bcent_det(pred_conf, target_conf) * self.params['balance'][i]

        # Final loss is a weighted sum of single losses
        loss_loc *= self.params['loc_w']
        loss_det *= self.params['det_w']
        loss_cls *= self.params['cls_w']

        loss = loss_loc + loss_det + loss_cls

        single_losses =  torch.tensor([loss_loc, loss_det, loss_cls], device=device)
        if reduction == 'sum':
            return loss, single_losses
        else:
            # Scale loss by batch size so that we have comparable values with v5
            return loss * batch_size, single_losses
