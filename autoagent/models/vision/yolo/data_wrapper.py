import random
import torch
import numpy as np
import cv2

from autoagent.utils.vision import compute_iou_centered
from autoagent.data.vision.augment import Resize, DataAugmentSeq


class YoloDataset(torch.utils.data.Dataset):
    """
    Ad-hoc map-style dataset for yolo.
    """
    def __init__(self, dataset, params, augment=True):
        self.dataset = dataset
        self.augment = augment
        self.params = params

    def __len__(self):
        return self.dataset.size

    def get_worker_init_fn(self):
        def worker_init_fn(id):
            wseed = id + random.randint(0, 2**16-1)
            random.seed(wseed)
            np.random.seed(wseed)
            torch.manual_seed(wseed)
        return worker_init_fn

    def __getitem__(self, idx):
        idx, img_dim = idx  # assuming we use MultiScaleBatchSampler
        img, target = self.dataset[idx]
        bboxes, classes = self.parse_target(target)
        size_augment = Resize((img_dim, img_dim))
        augments = self.params['augments'] if self.augment else []

        if augments and np.random.rand() < self.params['mosaic_prob']:
            # Mosaic augmentation
            # Pick 3 random images + this one
            indices = [*(np.random.randint(0, len(self)) for _ in range(3)), idx]
            four_imgs_and_targets = [self.dataset[i] for i in indices]
            four_imgs = [el[0] for el in four_imgs_and_targets]
            four_targets = [el[1] for el in four_imgs_and_targets]
            four_targets = [self.parse_target(four_targets[i]) for i in range(len(four_targets))]
            four_bboxes = [target[0] for target in four_targets]
            four_classes = [target[1] for target in four_targets]

            # Augment mosaic pieces
            a_four_imgs = []
            a_four_bboxes = []
            a_four_classes = []
            for i in range(4):
                # Standard augments
                a_img, a_bboxes = apply_aug_seq(
                    four_imgs[i], four_bboxes[i],
                    DataAugmentSeq(
                        transforms=augments,
                        probs=[0.5 for _ in range(len(augments))]
                    )
                )
                a_four_imgs.append(a_img)
                a_four_bboxes.append(a_bboxes)
                a_four_classes.append(four_classes[i])

            # Build mosaic
            f_img, f_bboxes, f_classes = mosaic(
                (img_dim, img_dim),
                a_four_imgs,
                a_four_bboxes,
                a_four_classes,
                self.params['mosaic_scale'],
                self.params['mosaic_translate']
            )

        else:
            if augments:
                # Standard augments
                f_img, f_bboxes = apply_aug_seq(
                    img, bboxes,
                    augment_seq=DataAugmentSeq(
                        transforms=[size_augment, *augments],
                        probs=[1, *[0.5 for _ in range(len(augments))]]
                    )
                )
                # Additional augments for obj detection
                f_img, f_bboxes = apply_random_crop(f_img, f_bboxes)
            else:
                # Just resize
                f_img, f_bboxes = apply_aug_seq(
                    img, bboxes,
                    augment_seq= DataAugmentSeq(
                        transforms=[size_augment],
                        probs=[1]
                    )
                )
            f_classes = classes

        return f_img, f_bboxes, f_classes

    def parse_target(self, target):
        """
        Parse VOC like dict target to retrieve bboxes and classes
        """
        classes = []
        bboxes = []

        for obj in target['objects']:
            classes.append(self.params['name_to_idx'][obj['name']])
            bbox = obj['bndbox']
            bboxes.append([
                int(bbox['xmin']),
                int(bbox['ymin']),
                int(bbox['xmax']),
                int(bbox['ymax'])
            ])
        return bboxes, classes

    def collate_fn(self, data):
        final_imgs = [x[0] for x in data]
        final_bboxes = [x[1] for x in data]
        final_classes = [x[2] for x in data]
        img_dim = final_imgs[0].shape[0]

        # Preprocess images
        batch_x = torch.cat([
            preprocess_x(
                final_imgs[i],
            ) for i in range(len(final_imgs))
        ])

        # Preprocess targets
        batch_y = []
        mode = self.params['mode']
        if mode == 'mult':
            thresh = self.params['mult_thresh']
        elif mode == 'iou':
            thresh = self.params['iou_thresh']

        for pre_idx in range(3):
            pre_list = [preprocess_y(
                i,
                final_bboxes[i],
                final_classes[i],
                img_dim,
                3,
                self.params['anchor_priors'][2-pre_idx],
                self.params['steps'][pre_idx],
                mode,
                thresh[pre_idx],
                self.params['num_cls']
            ) for i in range(len(final_imgs))]

            batch_y.append([
                torch.cat([el[j] for el in pre_list]) for j in range(5)
            ])

        info = {
            'bboxes': final_bboxes,
            'classes': final_classes
        }

        return batch_x, batch_y, info


def preprocess_x(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(x).float().permute(2, 0, 1)
    x = x / 255
    return x.unsqueeze(0)


def preprocess_y(img_idx, bboxes, classes, img_dim, num_anchors,
                 anchor_priors, step, mode, thresh, num_classes):
    grid_size = img_dim // step

    # Compute target tx, ty, w, h
    bboxes = torch.tensor([[*bboxes[i]] for i in range(len(classes))], dtype=torch.float32)
    if len(bboxes) == 0:
        target = torch.zeros((grid_size, grid_size, num_anchors, 5+num_classes), dtype=torch.float32)
        img_idx, anchor_idx = torch.LongTensor([]), torch.LongTensor([])
        x_idx, y_idx = torch.LongTensor([]), torch.LongTensor([])
    else:
        scaled_cx, scaled_cy = (bboxes[:, 2] + bboxes[:, 0])/2 , (bboxes[:, 3] + bboxes[:, 1])/2
        scaled_cx, scaled_cy = scaled_cx / (img_dim/grid_size), scaled_cy / (img_dim/grid_size)
        x, y = scaled_cx.long(), scaled_cy.long()
        tx, ty = scaled_cx - x, scaled_cy - y
        w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        w, h = w/(img_dim/grid_size), h / (img_dim/grid_size)
        scaled_anchor_priors = torch.from_numpy(anchor_priors) /(img_dim/grid_size)

        bboxes_wh = torch.zeros((w.shape[0], 2))
        bboxes_wh[:, 0] = w
        bboxes_wh[:, 1] = h

        # Find best anchors according to given threshold
        # Using multiple anchors (yolov4)
        if mode == 'iou':
            ious = torch.stack([compute_iou_centered(scaled_anchor.unsqueeze(0), bboxes_wh)
                                for scaled_anchor in scaled_anchor_priors]).t()
            ious = torch.where(ious > thresh, torch.tensor(1), torch.tensor(0))
            idx0, idx1 = torch.nonzero(ious, as_tuple=True)
        elif mode == 'mult':
            mults = torch.stack([torch.max(bboxes_wh / scaled_anchor.unsqueeze(0), scaled_anchor.unsqueeze(0) / bboxes_wh).max(dim=1)[0]
                                 for scaled_anchor in scaled_anchor_priors]).t()
            mults = torch.where(mults < thresh, torch.tensor(1), torch.tensor(0))
            idx0, idx1 = torch.nonzero(mults, as_tuple=True)

        # Obj indices
        x_idx = x[idx0].long()
        y_idx = y[idx0].long()
        anchor_idx = idx1.long()

        # Populate targets
        tx = tx[idx0]
        ty = ty[idx0]
        w = w[idx0]
        h = h[idx0]
        classes = torch.tensor(classes)[idx0]

        target = torch.zeros((grid_size, grid_size, num_anchors, 5+num_classes), dtype=torch.float32)

        target[y_idx, x_idx, anchor_idx, 0] = tx
        target[y_idx, x_idx, anchor_idx, 1] = ty
        target[y_idx, x_idx, anchor_idx, 2] = w
        target[y_idx, x_idx, anchor_idx, 3] = h
        target[y_idx, x_idx, anchor_idx, 4] = 1
        target[y_idx, x_idx, anchor_idx, 5+classes] = 1

        img_idx = torch.tensor([img_idx for _ in range(x_idx.shape[0])], dtype=torch.int64)

    # Return the target grid and target indices
    return target.unsqueeze(0), img_idx, x_idx, y_idx, anchor_idx


def apply_aug_seq(img, bboxes, augment_seq):
    img, t_funcs = augment_seq.augment(img)

    # Compute new bboxes after augmentations
    new_bboxes = []
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        for t_func in t_funcs:
            xmin, ymin = t_func(xmin, ymin)
            xmax, ymax = t_func(xmax, ymax)

        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        new_bboxes.append([round(xmin), round(ymin), round(xmax), round(ymax)])

    return img, new_bboxes


def random_crop(img, bboxes, pad_resize=True, pad=True):
    """
    Randomly crops img preserving all bounding boxes.
    If pad is True the cropped area is used as paddding.
    Otherwise we resize the non-cropped part to fill the original size (zoom).
    """
    h, w, c = img.shape

    xmin = min(bbox[0] for bbox in bboxes)
    xmax = max(bbox[2] for bbox in bboxes)
    ymin = min(bbox[1] for bbox in bboxes)
    ymax = max(bbox[3] for bbox in bboxes)

    d_left = round(np.random.rand() * (xmin))
    left = d_left

    d_right = round(np.random.rand() * (w - xmax))
    right = xmax + d_right

    d_top = round(np.random.rand() * (ymin))
    top = d_top

    d_bottom = round(np.random.rand() * (h - ymax))
    bottom = ymax + d_bottom

    if not pad:
        new_img = np.zeros((bottom-top, right-left, c), dtype=img.dtype)
        new_img[:] = img[top:bottom, left:right, :]
        r = Resize(img.shape[:-1], pad=pad_resize)
        new_img, t_func = r(new_img)

        new_bboxes = []
        for bbox in bboxes:
            new_bbox = [
                bbox[0] - left,
                bbox[1] - top,
                # remove 1, if xmax = w-1 -> right = xmax + 1
                bbox[2] - left - 1,
                # remove 1, if ymax = h-1 -> bottom = ymax + 1
                bbox[3] - top - 1
            ]
            xmin, ymin, xmax, ymax = new_bbox
            new_bbox[0], new_bbox[1] = t_func(xmin, ymin)
            new_bbox[2], new_bbox[3] = t_func(xmax, ymax)
            new_bboxes.append(new_bbox)
    else:
        new_img = np.zeros_like(img) + np.array([127, 127, 127], dtype=np.uint8)
        new_img[top:bottom, left:right, :] = img[top:bottom, left:right, :]
        new_bboxes = [bbox for bbox in bboxes]

    return new_img, new_bboxes


def apply_random_crop(img, bboxes):
    if np.random.rand() < 0.5:
        pad = True if np.random.rand() < 0.5 else False
        img, bboxes = random_crop(img, bboxes, pad=pad)
    return img, bboxes


def mosaic(mosaic_hw, imgs, bboxes, classes, scale, translate):
    """
    Mosaic augmentation.
    https://arxiv.org/pdf/2004.10934.pdf, Figure 3.
    """

    def valid_bbox(bbox, new_bbox, min_area=0.1):
        w1, h1 = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w2, h2 = new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]
        if (w2 > 2 and h2 > 2 and max(w2/h2, h2/w2) < 20
            and w1*h1 > 0 and (w2*h2)/(w1*h1) > min_area):
            return True
        else:
            return False


    def place_piece(mosaic, img, bboxes, classes, center, kind):
        """
        Place a single piece in the mosaic.
        """
        # Randomly scale mosaic piece
        r = Resize((mosaic.shape[0]//2, mosaic.shape[1]//2), pad=False)
        img, t_func = r(img)
        mos_h, mos_w, _ = mosaic.shape
        h, w ,_ = img.shape
        ch, cw = center

        if kind == 'tl':
            sw = max(w-cw, 0)
            sh = max(h-ch, 0)
            img_sw, img_ew = 0, w-sw
            img_sh, img_eh = 0, h-sh
            mos_sw, mos_ew = cw - (img_ew-img_sw), cw
            mos_sh, mos_eh = ch - (img_eh-img_sh), ch
            dw = cw - (img_ew-img_sw)
            dh = ch - (img_eh-img_sh)
        elif kind == 'tr':
            sw = max(w - (mos_w-cw), 0)
            sh = max(h-ch, 0)
            img_sw, img_ew = sw, w
            img_sh, img_eh = 0, h-sh
            mos_sw, mos_ew = cw, img_ew-img_sw + cw
            mos_sh, mos_eh = ch-(img_eh-img_sh), ch
            dw = cw - sw
            dh = ch - (img_eh-img_sh)
        elif kind == 'bl':
            sw = max(w-cw, 0)
            sh = max(h - (mos_h-ch), 0)
            img_sw, img_ew = 0, w-sw
            img_sh, img_eh = sh, h
            mos_sw, mos_ew = cw - (img_ew-img_sw), cw
            mos_sh, mos_eh = ch, img_eh-img_sh + ch
            dw = cw - (img_ew-img_sw)
            dh = ch - sh
        elif kind == 'br':
            sw = max(w - (mos_w-cw), 0)
            sh = max(h - (mos_h-ch), 0)
            img_sw, img_ew = sw, w
            img_sh, img_eh = sh, h
            mos_sw, mos_ew = cw, img_ew-img_sw + cw
            mos_sh, mos_eh = ch, img_eh-img_sh + ch
            dw = cw - sw
            dh = ch - sh

        mosaic[mos_sh:mos_eh, mos_sw:mos_ew, :] = img[img_sh:img_eh, img_sw:img_ew, :]

        new_bboxes = []
        new_classes = []
        for (xmin, ymin, xmax, ymax), c in zip(bboxes, classes):
            xmin, ymin = t_func(xmin, ymin)
            xmax, ymax = t_func(xmax, ymax)

            # bbox after resize
            bbox = [xmin, ymin, xmax, ymax]

            xmin += dw
            xmax += dw
            ymin += dh
            ymax += dh
            if xmin > mos_ew or ymin > mos_eh or xmax < mos_sw or ymax < mos_sh:
                # Bbox is out of bounds
                continue
            # Clip to be inside piece limits
            xmin = np.clip(xmin, mos_sw, mos_ew)
            xmax = np.clip(xmax, mos_sw, mos_ew)
            ymin = np.clip(ymin, mos_sh, mos_eh)
            ymax = np.clip(ymax, mos_sh, mos_eh)

            # new bbox
            new_bbox = [xmin, ymin, xmax, ymax]
            if valid_bbox(bbox, new_bbox):
                new_bboxes.append([xmin, ymin, xmax, ymax])
                new_classes.append(c)

        return new_bboxes, new_classes

    mosaic_hw = tuple(x*2 for x in mosaic_hw)
    mosaic = np.zeros(mosaic_hw + (3,), dtype=np.uint8) + 127
    w, h, _ = mosaic.shape
    center_h, center_w = [int(random.uniform(x, x*4 - x)) for x in [h/4, w/4]]
    center = (center_h, center_w)

    mosaic_bboxes = []
    mosaic_classes = []

    # Top-left, Top-right, Bottom-Left, Bottom-Right
    kinds = ['tl', 'tr', 'bl', 'br']
    for i in range(4):
        new_bboxes, new_classes = place_piece(mosaic, imgs[i], bboxes[i], classes[i], center, kinds[i])
        mosaic_bboxes.extend((i for i in new_bboxes))
        mosaic_classes.extend((i for i in new_classes))

    # Random scale and translation
    # Move center to origin
    m1 = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Scale
    s = np.random.uniform(1-scale, 1+scale)
    m2 = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Translate
    tx = np.random.uniform(0.5 - translate, 0.5 + translate) * w/2
    ty = np.random.uniform(0.5 - translate, 0.5 + translate) * h/2

    m3 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    # Compose
    m = m3.dot(m2.dot(m1))

    # Target size
    h, w = h//2, w//2

    mosaic = cv2.warpAffine(mosaic, m[:2], (w, h), borderValue=(127,127,127))

    def t_func(x, y):
        return m.dot(np.array([x, y, 1]))[:2]

    final_bboxes = []
    final_classes = []
    for i, (xmin, ymin, xmax, ymax) in enumerate(mosaic_bboxes):
        bbox = [xmin, ymin, xmax, ymax]
        bbox = [x*s for x in bbox]  # scale for valid check
        xmin, ymin = t_func(xmin, ymin)
        xmax, ymax = t_func(xmax, ymax)
        if xmin > w or ymin > h or xmax < 0 or ymax < 0:
            # Bbox is out of bounds
            continue
        # Clip to be inside piece limits
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        new_bbox = [xmin, ymin, xmax, ymax]
        new_bbox = [int(round(x)) for x in new_bbox]
        if valid_bbox(bbox, new_bbox):
            final_bboxes.append(new_bbox)
            final_classes.append(mosaic_classes[i])

    return mosaic, final_bboxes, final_classes
