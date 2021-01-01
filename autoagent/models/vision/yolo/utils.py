import torch
import torchvision

from typing import List, Tuple


@torch.jit.script
def compute_final_bboxes(preds: List[torch.Tensor], conf_thresh: float, anchor_priors: List[torch.Tensor],
                         auxiliary_grids: List[torch.Tensor], steps: List[torch.Tensor],
                         with_ids: bool, bbox_fn: str) -> torch.Tensor:
    """
    Compute (ready to render - before nms) bounding boxes from a yolo prediction grid.
    Each bounding box has the form:
        (xmin, ymin, xmax, ymax, confidence, class, class_prob, Optional(img_id))
    All bounding boxes below a given confidence threshold are discarded.
    """
    device = preds[0].device
    batch_size = preds[0].shape[0]
    bboxes: List[torch.Tensor] = []
    for i in range(len(preds)):
        pred = preds[i]
        gy, gx = pred.shape[2:4]
        num_anchors = pred.shape[1]
        stepxy = steps[i]

        pred[..., 0:2] = 2*torch.sigmoid(pred[..., 0:2]) - 0.5 + auxiliary_grids[i]

        if bbox_fn == 'exp':
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * (anchor_priors[i] / stepxy)
        else:
            pred[..., 2:4] = (2*torch.sigmoid(pred[..., 2:4]))**2 * (anchor_priors[i] / stepxy)
        pred[..., 4:] = torch.sigmoid(pred[..., 4:])

        if with_ids:
            id = torch.zeros((batch_size, num_anchors, gy, gx, 1)).to(device)
            for i in range(batch_size):
                id[i, ...] = i
            pred = torch.cat([pred, id], -1)
            start: int = 5
            end: int = -1
        else:
            start: int = 5
            end: int = pred.shape[-1]

        # Filter by overall confidence
        best_bboxes_mask = pred[..., 4] * torch.max(pred[..., start:end], dim=-1)[0] > conf_thresh
        pred_filtered = pred[best_bboxes_mask]
        bboxes_xywh = pred_filtered[:, :4]

        if bboxes_xywh.size()[0] != 0:
            bboxes_xyxy = torch.zeros_like(bboxes_xywh)
            bboxes_xyxy[:, 0] = (bboxes_xywh[:, 0] - bboxes_xywh[:, 2]/2) * stepxy[0]
            bboxes_xyxy[:, 1] = (bboxes_xywh[:, 1] - bboxes_xywh[:, 3]/2) * stepxy[1]
            bboxes_xyxy[:, 2] = (bboxes_xywh[:, 0] + bboxes_xywh[:, 2]/2) * stepxy[0]
            bboxes_xyxy[:, 3] = (bboxes_xywh[:, 1] + bboxes_xywh[:, 3]/2) * stepxy[1]

            # Add confidence, class, class_prob, and id
            probs, classes = torch.max(pred_filtered[:, start:end], dim=1, keepdim=True)
            confidences = pred_filtered[:, 4].unsqueeze(1)
            if with_ids:
                ids = pred_filtered[:, -1].unsqueeze(1)
                pred_bboxes = torch.cat([bboxes_xyxy, confidences, classes, probs, ids], dim=1)
            else:
                pred_bboxes = torch.cat([bboxes_xyxy, confidences, classes, probs], dim=1)

        else:
            pred_bboxes = torch.empty(0, device=device)

        bboxes.append(pred_bboxes)

    return torch.cat(bboxes, dim=0)


@torch.jit.script
def non_max_suppression(bboxes, iou_thresh: float = 0.4, per_class: int = 1,
                        n_max: int = 256, merge_conf: bool = True):
    """
    Filtering bboxes via non-max suppression.
    """
    bboxes = bboxes.float()
    if merge_conf:
        # Conf = conf * cls_conf
        bboxes[:, 4] *= bboxes[:, 6]
    # Offset bboxes to "separate" classes for nms
    off_bboxes = bboxes[:, :4] + (bboxes[:, 5:6] * 4096) * per_class
    indices = torchvision.ops.nms(off_bboxes, bboxes[:, 4], iou_thresh)
    return bboxes[indices[:n_max]][:, :6]
