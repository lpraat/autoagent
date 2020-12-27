import unittest
import numpy as np
import torch

from autoagent.utils.vision_utils import compute_iou, compute_iou_centered, compute_generalized_iou

class TestYolo(unittest.TestCase):

    def test_compute_iou(self):

        # Max ious
        bboxes1 = np.array([
            [0, 0, 20, 20],
            [10, 10, 30, 30],
            [-10, -5, 10, 5]
        ], dtype=np.float32)
        bboxes2 = np.copy(bboxes1)

        ious = compute_iou(torch.from_numpy(bboxes1),
                           torch.from_numpy(bboxes2))

        np.testing.assert_array_almost_equal(ious, 1)

        # Min ious
        bboxes2 = np.array([
            [20, 20, 30, 30],
            [-1, -1, 0, 0],
            [-30, -30, -20, -20]
        ], dtype=np.float32)

        ious = compute_iou(torch.from_numpy(bboxes1),
                           torch.from_numpy(bboxes2))

        np.testing.assert_array_almost_equal(ious, 0)

        # Ious
        bboxes2 = np.array([
            [10, 10, 20, 20],
            [15, 15, 25, 25],
            [-10, -5, 20, 5]
        ], dtype=np.float32)


        ious = compute_iou(
            torch.from_numpy(bboxes1),
            torch.from_numpy(bboxes2)
        )
        np.testing.assert_array_almost_equal(
            ious,
            np.array([0.25, 0.25, 2/3])
        )

    def test_compute_iou_centered(self):
        bboxes1 = np.array([
            [50, 50],
            [0, 0],
            [300, 300],
        ], dtype=np.float32)

        bboxes2 = np.array([
            [100, 100],
            [200, 200],
            [50, 50]
        ], dtype=np.float32)

        ious = compute_iou_centered(
            torch.from_numpy(bboxes1),
            torch.from_numpy(bboxes2)
        )

        np.testing.assert_array_almost_equal(
            ious,
            np.array([
                (50*50)/(100*100),
                0,
                (50*50)/(300*300)
            ])
        )

    def test_generalized_iou(self):
        bboxes1 = torch.from_numpy(np.array([
            [0, 0, 20, 20],
            [10, 10, 30, 30],
            [-10, -5, 10, 5]
        ], dtype=np.float32))

        bboxes2 = torch.from_numpy(np.array([
            [-10, 10, 20, 20],
            [15, 15, 25, 25],
            [-10, -5, 20, 5]
        ], dtype=np.float32))

        dious = compute_generalized_iou(bboxes1, bboxes2, kind='diou')
        np.testing.assert_array_almost_equal(
            dious,
            np.array([
                0.36153847,
                0.2500,
                0.6416667
            ])
        )

        cious = compute_generalized_iou(bboxes1, bboxes2, kind='ciou')
        np.testing.assert_array_almost_equal(
            cious,
            np.array([
                0.35049164,
                0.2500,
                0.6414717
            ])
        )

