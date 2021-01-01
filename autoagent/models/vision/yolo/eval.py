import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from autoagent.models.vision.yolo.utils import compute_final_bboxes, non_max_suppression
from autoagent.utils.obj_det_metrics import compute_ap
from autoagent.utils.general import fancy_float


def get_prcurves(precs, recs, classes):
    figs = []
    for i, cls in enumerate(classes):
        p, r = precs[i], recs[i]
        new_fig = plt.figure(num=cls)
        figs.append(new_fig)

        plt.title(cls)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(r, p)

    return figs


def eval(yolo, dloader, batch_size, aggregate, epoch, confidence_thresh, nms_thresh,
         name='Val'):
    """
    Evaluates yolo model against eval data from dloader.
    """
    yolo.eval()
    eval_single_losses = 0  # running mean
    single_losses = 0  # tmp aggregate
    params = yolo.params

    ground_truths = dict()
    detections = []
    classes = set()

    # Progress bar
    iter_per_epoch = len(dloader.dataset)//(batch_size*aggregate)
    titles = ['AP@0.5', f'mAP@0.5:0.95', 'BatchNum', 'Loss', 'LocLoss', 'DetLoss', 'ClsLoss', 'Precision', 'Recall']
    s_titles = "".join(f"{t:<15}" for t in titles)
    print(s_titles)
    pbar = tqdm.tqdm(total=iter_per_epoch, dynamic_ncols=True)

    for eval_iter, (x, y, info) in enumerate(dloader):

        # Populate ground truths
        for i, img_bboxes in enumerate(info['bboxes']):
            # i*(eval_iter+batch_size) is the unique identifier for an image
            ground_truths[i+(eval_iter*batch_size)] = []
            for xyxy, cls in zip(img_bboxes, info['classes'][i]):
                ground_truths[i+(eval_iter*batch_size)].append([*xyxy, cls])
                if cls not in classes:
                    classes.add(cls)

        # Predict
        with torch.no_grad():
            x = x.cuda(non_blocking=dloader.pin_memory)
            y = [[el.cuda(non_blocking=dloader.pin_memory) for el in t] for t in y]

            preds = yolo(x)
            loss, _single_losses = yolo.get_loss(preds, y)

        # Retrieve also the ids to associate each bbox to the correct image
        bboxes = yolo.detect(x, confidence_thresh, with_ids=True)

        if bboxes.shape[0] > 0:
            # For each image compute final detections
            ids = torch.unique(bboxes[..., -1].int()).tolist()
            for id in ids:
                img_bboxes = bboxes[bboxes[:, -1].int() == id][:, :-1]

                if img_bboxes.shape[0] > 0:
                    img_bboxes = non_max_suppression(img_bboxes, iou_thresh=nms_thresh)

                    for img_bbox in img_bboxes:
                        # [img_id, xmin, ymin, xmax, ymax, conf, cls]
                        new_det = [id+(eval_iter*batch_size), *img_bbox.tolist()]
                        new_det[-1] = int(new_det[-1])
                        detections.append(new_det)

        # Update eval losses
        single_losses += _single_losses / aggregate

        if (eval_iter+1) % aggregate == 0:
            iter_num = (eval_iter+1)//aggregate

            eval_single_losses = (eval_single_losses * (iter_num-1) + single_losses) / iter_num

            descs = [
                " ", " ",
                f"{iter_num}/{iter_per_epoch}", f"{fancy_float(torch.sum(eval_single_losses).item())}",
                *(f"{fancy_float(eval_single_losses[i].item())}" for i in range(3)),
                " ", " "
            ]
            s_descs = "".join(f"{d:<15}" for d in descs)
            pbar.set_description(s_descs)

            single_losses = 0
            pbar.update()

    # Use ground truths and detections to compute Precision, Recall and AP
    ps = []
    rs = []
    aps = []
    pr_pointss = []
    classes = [i for i in sorted(list(classes))]
    for i, thresh in enumerate(np.arange(0.5, 1, step=0.05)):
        prcurve = True if i == 0 else False
        p, r, ap, pr_points = compute_ap(ground_truths, detections, classes, iou_thresh=thresh, prcurve=prcurve)
        ps.append(p)
        rs.append(r)
        aps.append(ap)
        pr_pointss.append(pr_points)

    p = np.mean(ps[0])
    r = np.mean(rs[0])
    f1 = 2*(p * r)/(p + r + 1e-7)

    ap_05 = aps[0].mean()
    ap_095 = np.array(aps).mean()

    named_classes = [yolo.params['idx_to_name'][i] for i in classes]
    pr_curves = get_prcurves([t[0] for t in pr_pointss[0]], [t[1] for t in pr_pointss[0]], named_classes)

    descs = [
        f"{fancy_float(ap_05)}", f"{fancy_float(ap_095)}",
        f"{iter_num}/{iter_per_epoch}", f"{fancy_float(torch.sum(eval_single_losses).item())}",
        *(f"{fancy_float(eval_single_losses[i].item())}" for i in range(3)),
        f"{fancy_float(p)}", f"{fancy_float(r)}"
    ]
    s_descs = "".join(f"{d:<15}" for d in descs)
    pbar.set_description(s_descs)

    statistics = {
        'Loss': torch.sum(eval_single_losses).item(),
        'Loc_loss': eval_single_losses[0].item(),
        'Det_loss': eval_single_losses[1].item(),
        'Cls_loss': eval_single_losses[2].item(),
        'Precision': p,
        'Recall': r,
        'F1': f1,
        'AP@0.5': ap_05,
        'AP@0.5:0.95': ap_095
    }

    pbar.close()
    return statistics, zip(named_classes, pr_curves)
