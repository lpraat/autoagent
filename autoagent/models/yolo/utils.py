import torch
import numpy as np
import cv2
import torchvision

from autoagent.utils.vision_utils import compute_iou, compute_generalized_iou


def compute_final_bboxes(img_dims, num_classes, anchor_priors, pred_grid, bbox_fn,
                         from_preds=True, confidence_thresh=0.5, with_ids=False,
                         device='cuda'):
    """
    Compute (ready to render) bounding boxes from a yolo prediction grid.
    Each bounding box has the form:
        (xmin, ymin, xmax, ymax, confidence, class, class_prob, img_id)
    All bounding boxes below a given confidence threshold are discarded.
    """
    batch_size = pred_grid.shape[0]
    gy, gx = pred_grid.shape[2:4]
    img_h, img_w = img_dims
    stepy, stepx = img_h / gy, img_w / gx
    num_anchors = anchor_priors.shape[0]

    # Prepare auxiliary reconstruction grid
    grid_id = torch.zeros((batch_size, gy, gx, num_anchors, 1), device=device)
    grid_x = torch.zeros((batch_size, gy, gx, num_anchors, 1), device=device)
    grid_y = torch.zeros((batch_size, gy, gx, num_anchors, 1), device=device)
    grid_pw = torch.zeros((batch_size, gy, gx, num_anchors, 1), device=device)
    grid_ph = torch.zeros((batch_size, gy, gx, num_anchors, 1), device=device)
    grid_fill = torch.zeros((batch_size, gy, gx, num_anchors, 1+num_classes), device=device)

    # Id is the batch index
    for i in range(batch_size):
        grid_id[i, ...] = i

    grid_x[..., 0] = torch.stack(
            [torch.tensor([i for i in range(gx)]) for _ in range(gy)]
        ).reshape(gy, gx, 1)
    grid_y[..., 0] = torch.stack(
            [torch.tensor([i for i in range(gy)]) for _ in range(gx)]
        ).t().reshape(gy, gx, 1)

    grid_pw[..., 0] = anchor_priors[:, 0] / stepx
    grid_ph[..., 0] = anchor_priors[:, 1] / stepy

    grid = torch.cat([grid_x, grid_y, grid_pw, grid_ph, grid_fill, grid_id], dim=-1)
    grid = grid.permute(0, 3, 1, 2, 4).contiguous()

    # Build output grid
    final_grid = torch.zeros_like(grid)

    if from_preds:
        final_grid[..., 0] = grid[..., 0] + 2 * torch.sigmoid(pred_grid[..., 0]) - 0.5
        final_grid[..., 1] = grid[..., 1] + 2 * torch.sigmoid(pred_grid[..., 1]) - 0.5
    else:
        final_grid[..., 0] = grid[..., 0] + 2 * pred_grid[..., 0] - 0.5
        final_grid[..., 1] = grid[..., 1] + 2 * pred_grid[..., 1] - 0.5

    if bbox_fn == 'exp':
        final_grid[..., 2] = grid[..., 2] * torch.exp(pred_grid[..., 2])
        final_grid[..., 3] = grid[..., 3] * torch.exp(pred_grid[..., 3])
    else:
        final_grid[..., 2] = grid[..., 2] * ((torch.sigmoid(pred_grid[..., 2]) * 2) ** 2)
        final_grid[..., 3] = grid[..., 3] * ((torch.sigmoid(pred_grid[..., 3]) * 2) ** 2)

    if from_preds:
        final_grid[..., 4] = torch.sigmoid(pred_grid[..., 4])
        final_grid[..., 5:-1] = torch.sigmoid(pred_grid[..., 5:])
    else:
        final_grid[..., 4] = pred_grid[..., 4]
        final_grid[..., 5:-1] = pred_grid[..., 5:]

    final_grid[..., -1] = grid[..., -1]

    # Build (xmin, ymin, xmax, ymax) for each bbox
    # conf = obj_conf * class_p
    best_bboxes_mask = final_grid[..., 4] * torch.max(final_grid[..., 5:-1], dim=-1)[0] > confidence_thresh
    bboxes_xywh = final_grid[best_bboxes_mask][:, :4]

    if bboxes_xywh.size()[0] != 0:
        bboxes_xyxy = torch.zeros_like(bboxes_xywh, dtype=torch.float32, device=device)

        bboxes_xyxy[:, 0] = bboxes_xywh[:, 0] * stepx - bboxes_xywh[:, 2]*stepx/2
        bboxes_xyxy[:, 1] = bboxes_xywh[:, 1] * stepy - bboxes_xywh[:, 3]*stepy/2
        bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] * stepx + bboxes_xywh[:, 2]*stepx/2
        bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] * stepy + bboxes_xywh[:, 3]*stepy/2

        # Add confidence, class, class_prob, and id
        probs, classes = torch.max(final_grid[best_bboxes_mask][:, 5:-1], dim=1, keepdim=True)
        confidences = final_grid[best_bboxes_mask][:, 4].unsqueeze(1)
        if with_ids:
            ids = final_grid[best_bboxes_mask][:, -1].unsqueeze(1)
            return torch.cat([bboxes_xyxy, confidences, classes, probs, ids], dim=1)
        else:
            return torch.cat([bboxes_xyxy, confidences, classes, probs], dim=1)

    else:
        return torch.tensor([], device=device)


def non_max_suppression(bboxes, iou_thresh=0.4, per_class=1, n_max=256):
    """
    Filtering bboxes via non-max suppression.
    """
    # Offset bboxes to "separate" classes for nms
    off_bboxes = bboxes[:, :4] + (bboxes[:, 5:6] * 4096) * per_class
    scores = bboxes[:, 4] * bboxes[:, 6]  # conf * cls_conf
    indices = torchvision.ops.nms(off_bboxes, scores, iou_thresh)
    return bboxes[indices[:n_max]]


# SLOW diou nms, TODO make it fast
# def non_max_suppression(bboxes, iou_thresh=0.4, per_class=1, n_max=1000, kind='diou'):
#     """
#     Filtering bboxes via non-max suppression.
#     """
#     final_bboxes = []

#     bboxes[:, :4] += (bboxes[:, 5:6] * 5e3) * per_class

#     while bboxes.shape[0] > 0:
#         _, max_index = torch.max(bboxes[:, 4], dim=0)
#         max_box = bboxes[max_index]
#         final_bboxes.append(max_box)

#         remaining_bboxes = bboxes[[i for i in range(bboxes.shape[0]) if i!=max_index]]
#         ious = compute_generalized_iou(max_box[:4].unsqueeze(0), remaining_bboxes[...,:4], kind=kind)
#         indices_to_keep = ious < iou_thresh

#         bboxes = remaining_bboxes[indices_to_keep]

#     final_bboxes = torch.stack(final_bboxes)
#     final_bboxes[:, :4] -= (final_bboxes[:, 5:6] * 5e3) * per_class

#     return final_bboxes
