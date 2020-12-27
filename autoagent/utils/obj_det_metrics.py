import numpy as np


def compute_iou(bbox1, bbox2, eps=1e-7):
    xmin1, ymin1, xmax1, ymax1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    xmin2, ymin2, xmax2, ymax2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    xmin = max(xmin1, xmin2)
    xmax = min(xmax1, xmax2)
    ymin = max(ymin1, ymin2)
    ymax = min(ymax1, ymax2)

    intersection_area = max(xmax - xmin + 1, 0) * max(ymax - ymin + 1, 0)
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    return intersection_area / ((area1 + area2) - intersection_area + eps)


def compute_ap_per_cls(ground_truths, detections, iou_thresh=0.5):
    already_found_tp = set()

    gt_size = sum(len(gt) for gt in ground_truths.values())

    conf, prec, rec = [], [], []
    acc_tp, acc_fp = 0, 0

    ap_rec = [0]
    ap_sampling_indices = []

    for det in detections:
        img = det[0]
        bbox = det[1:5]
        conf.append(det[5])

        gts = ground_truths[img]

        iou_max = 0
        target_gt = None

        for i, gt in enumerate(gts):
            iou = compute_iou(bbox, gt)

            if iou > iou_max:
                iou_max = iou
                target_gt = i

        if iou_max >= iou_thresh and (img, target_gt) not in already_found_tp:
            # True positive
            acc_tp += 1
            already_found_tp.add((img, target_gt))
        else:
            # False positive
            acc_fp += 1

        rec.append(acc_tp / gt_size)
        if len(rec) == 1 or rec[-1] != rec[-2]:
            ap_rec.append(rec[-1])
            # Sample precision values at different recall values
            ap_sampling_indices.append(len(rec)-1)

        prec.append(acc_tp / (acc_tp + acc_fp))

    if len(prec) == 0:
        return 0, 0, 0, [], []

    # Sampled precisions
    ap_prec = [p for p in prec]
    for i in range(len(prec)-2, 0, -1):
        ap_prec[i] = max(prec[i], ap_prec[i+1])
    ap_prec = [ap_prec[i] for i in ap_sampling_indices]

    # Compute AP@iou_thresh
    ap = 0
    for i in range(len(ap_rec)-1):
        ap += (ap_rec[i+1] - ap_rec[i])*ap_prec[i]

    # While entry confidence for mAP is usually 0.001
    # get precision and recall at 0.1
    p = np.interp(-0.1, -np.array(conf), np.array(prec))
    r = np.interp(-0.1, -np.array(conf), np.array(rec))
    return ap, p, r, prec, rec


def compute_ap(ground_truths, detections, classes, iou_thresh=0.5, prcurve=False):
    """Compute AP@iou_thresh as described in
        https://github.com/rafaelpadilla/Object-Detection-Metrics

    Parameters
    ----------
    ground_truths : dict
        Maps each img id to the list of ground truth bboxes.
        Each bbox has the form [xmin, ymin, xmax, ymax, cls].
    detections : list
        List of detected bboxes.
        Each bbox has the form [img_id, xmin, ymin, xmax, ymax, conf, cls].
    classes : list
        Class ids.
    iou_thresh : float, optional
        Threshold, by default 0.5

    Returns
    -------
    tuple
        Precision, recall, AP, and the points to plot a precision-recall curve
        for each class.
    """
    # Sort detections by confidence
    detections.sort(key=lambda x: -x[-2])

    p = np.zeros(len(classes), dtype=np.float32)
    r = np.zeros_like(p, dtype=np.float32)
    ap = np.zeros_like(p, dtype=np.float32)
    pr_points = []

    # Compute metrics for each class
    for i, cls in enumerate(classes):
        cls_gts = {}
        for img in ground_truths.keys():
            gts = ground_truths[img]
            cls_gts[img] = [gt for gt in gts if gt[-1] == cls]

        cls_ap, cls_p, cls_r, cls_ppts, cls_rpts = compute_ap_per_cls(
            cls_gts,
            [det for det in detections if det[-1] == cls],
            iou_thresh
        )

        p[i] = cls_p
        r[i] = cls_r
        ap[i] = cls_ap

        if prcurve:
            pr_points.append([cls_ppts, cls_rpts])

    return p, r, ap, pr_points