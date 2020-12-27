import math
import torch
import numpy as np


def get_dim_after_conv_2d(dim, padding, filter, stride):
    return math.floor((dim + 2*padding - filter) / stride + 1)


def compute_iou_centered(bbox1, bbox2):
    w1, h1 = bbox1[:, 0], bbox1[:, 1]
    w2, h2 = bbox2[:, 0], bbox2[:, 1]

    intersection_area = torch.min(w1, w2) *  torch.min(h1, h2)
    return intersection_area / (w1*h1 + w2*h2 - intersection_area)


def compute_iou(bbox1, bbox2, eps=1e-7):
    xmin1, ymin1, xmax1, ymax1 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    xmin2, ymin2, xmax2, ymax2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    xmin = torch.max(xmin1, xmin2)
    xmax = torch.min(xmax1, xmax2)
    ymin = torch.max(ymin1, ymin2)
    ymax = torch.min(ymax1, ymax2)

    intersection_area = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    return intersection_area / ((area1 + area2) - intersection_area + eps)


def compute_generalized_iou(bbox1, bbox2, format='xyxy', kind='ciou', eps=1e-7):
    if format == 'xyxy':
        xmin1, ymin1, xmax1, ymax1 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
        xmin2, ymin2, xmax2, ymax2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]
    elif format == 'xywh':
        xmin1, xmin2 = bbox1[:, 0] - bbox1[:, 2]/2, bbox2[:, 0] - bbox2[:, 2]/2
        xmax1, xmax2 = bbox1[:, 0] + bbox1[:, 2]/2, bbox2[:, 0] + bbox2[:, 2]/2
        ymin1, ymin2 = bbox1[:, 1] - bbox1[:, 3]/2, bbox2[:, 1] - bbox2[:, 3]/2
        ymax1, ymax2 = bbox1[:, 1] + bbox1[:, 3]/2, bbox2[:, 1] + bbox2[:, 3]/2

    xmin = torch.max(xmin1, xmin2)
    xmax = torch.min(xmax1, xmax2)
    ymin = torch.max(ymin1, ymin2)
    ymax = torch.min(ymax1, ymax2)

    intersection_area = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou = intersection_area / ((area1 + area2) - intersection_area + eps)

    # Central points
    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    cx1 = xmin1 + w1/2
    cy1 = ymin1 + h1/2
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2
    cx2 = xmin2 + w2/2
    cy2 = ymin2 + h2/2

    # Euclidean distance between central points
    rho2 = (cx1 - cx2)**2 + (cy1 - cy2)**2

    # Diagonal length of the smallest enclosing box covering two boxes
    c2 = (torch.max(xmax1, xmax2) - torch.min(xmin1, xmin2))**2 + (torch.max(ymax1, ymax2) - torch.min(ymin1, ymin2))**2

    # https://arxiv.org/pdf/1911.08287.pdf
    if kind == 'diou':
        penalty_term = rho2/(c2+eps)

    elif kind == 'ciou':
        v = 4/(np.pi**2) * (torch.atan(w2/(h2+eps)) - torch.atan(w1/(h1+eps)))**2
        with torch.no_grad():
            alpha = v / ((1-iou) + v + eps)
        penalty_term = rho2/(c2+eps) + alpha*v

    else:
        return compute_iou(bbox1, bbox2)

    return iou - penalty_term
