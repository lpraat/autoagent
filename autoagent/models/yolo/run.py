import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns

from typing import Tuple

from torch.serialization import save
from autoagent.data.vision.augment import Resize
from autoagent.models.yolo.model import Yolo
from autoagent.models.yolo.layers import Conv
from autoagent.models.yolo.config.parse_config import parse_params
from autoagent.models.yolo.utils import non_max_suppression
from autoagent.utils.torch_utils import get_num_params


colors = sns.color_palette("muted") + sns.color_palette("colorblind") + sns.color_palette("deep")
colors = [[min(int(x*255), 255) for x in el] for el in colors]
font_scale = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 1.2

def draw_bboxes(img, bboxes, idx_to_name):
    for xmin, ymin, xmax, ymax, confidence, c_idx in bboxes:
        c_idx = int(c_idx.item())
        color = colors[c_idx % len(colors)]
        img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, thickness=4)

        text = f"{idx_to_name[c_idx]}({confidence*100:.1f}%)"
        (txt_width, txt_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

        cv2.rectangle(img, (xmin, ymin), (xmin+txt_width, ymin-txt_height), color, cv2.FILLED)
        cv2.putText(img, text, (xmin, ymin), font, fontScale=0.8*font_scale, color=(255, 255, 255))
    return img


@torch.jit.script
def bbox_transform(bboxes: torch.Tensor, size: Tuple[int, int], t: Tuple[int, int], hw: Tuple[int, int]):
    bboxes[:, torch.tensor([0, 2])] -= (size[1] - t[1])/2
    bboxes[:, torch.tensor([1, 3])] -= (size[0] - t[0])/2
    bboxes[:, torch.tensor([0, 2])] *= hw[1] / t[1]
    bboxes[:, torch.tensor([1, 3])] *= hw[0] / t[0]
    return bboxes


def preprocess_img(img, img_dim, mult=32):
    h, w, _ = img.shape
    ratio = min(img_dim/w, img_dim/h)

    # Target width and height
    t_w = int(w*ratio)
    t_h = int(h*ratio)

    # Full size (with pad)
    size_h, size_w = math.ceil(t_h / mult)*32, math.ceil(t_w / mult)*32
    img = Resize(size=(size_h, size_w), pad=True)(img)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255
    return img, (size_h, size_w), (t_h, t_w), (h, w)


def run_on_img(model, orig_img, img_dim, half_precision, cuda, conf_thresh, nms_thresh, use_cache):
    img, size, t, hw = preprocess_img(orig_img, img_dim)
    if half_precision:
        img = img.half()
    if cuda:
        img = img.cuda()

    bboxes = model.detect(img, conf_thresh=conf_thresh, use_cache=use_cache)
    if bboxes.shape[0] > 0:
        bboxes = bbox_transform(
            non_max_suppression(bboxes, iou_thresh=nms_thresh),
            size, t, hw
        ).cpu()
    f_img = draw_bboxes(orig_img, bboxes, model.params['idx_to_name'])
    return f_img


def run(model, path, img_dim, conf_thresh, nms_thresh, half_precision, cuda, view_res, save_path):
    model.model = torch.jit.script(model.model)

    model.eval()
    if half_precision:
        model.half()
    if cuda:
        model.cuda()

    img_exts = set(('.jpeg', '.jpg', '.png'))

    if os.path.splitext(path)[1] in img_exts:
        orig_img = cv2.imread(path)
        res = run_on_img(model, orig_img, img_dim, half_precision, cuda, conf_thresh, nms_thresh, use_cache=True)
        if view_res:
            cv2.imshow("Result", res)
            cv2.waitKey()
        if save_path is not None:
            cv2.imwrite(save_path, res)
    else:
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(path)
        if save_path:
            out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, four_cc, cap.get(cv2.CAP_PROP_FPS), (out_w, out_h))

        quit_key = ord('q')

        curr = time.perf_counter()
        frames = 0
        fps = 0
        while (cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break

            res = run_on_img(model, frame, img_dim, half_precision, cuda, conf_thresh, nms_thresh, use_cache=True)

            if view_res:
                cv2.imshow('Result', res)

            if save_path:
                out.write(res)

            frames += 1
            if time.perf_counter() - curr > 1:
                curr = time.perf_counter()
                fps = frames
                frames = 0

            if cv2.waitKey(1) == quit_key:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("YoloRun")

    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint with trained weights')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to params config file')
    parser.add_argument('--source', type=str, required=True,
                        help='Source file')
    parser.add_argument('--img_dim', type=int, required=True,
                        help='Img dimension for inference')
    parser.add_argument('--conf_thresh', type=float, required=True,
                        help='Minimum bbox acceptance threshold')
    parser.add_argument('--nms_thresh', type=float, required=True,
                        help='Non maximum supp. threshold')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision')
    parser.add_argument('--cuda', action='store_true',
                        help='Use gpu acceleration')
    parser.add_argument('--view_res', action='store_true',
                        help='View the results')
    parser.add_argument('--save_path', default=None,
                        help='Output path, where the results are saved')
    args = parser.parse_args()

    params = parse_params(args.params)[1]
    yolo = Yolo(params)
    yolo.load_state_dict(torch.load(args.ckpt))

    run(yolo, args.source, args.img_dim, args.conf_thresh, args.nms_thresh,
        args.half_precision, args.cuda, args.view_res, args.save_path)
