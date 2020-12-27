import unittest
import numpy as np
import torch

from autoagent.models.yolo.utils import non_max_suppression

class TestYoloUtils(unittest.TestCase):

    def test_non_max_suppression(self):
        bboxes = torch.tensor([
            [20, 20, 50, 50, 1.0, 9, 0.7],
            [50, 50, 100, 100, 1.0, 8, 0.9],
            [20, 20, 45, 45, 0.8, 7, 1],
            [20, 20, 49, 49, 0.7, 6, 0],
            [50, 50, 90, 90, 0.7, 5, 0],
            [-20, -20, 0, 0, 0.9, 4, 0]
        ])

        target = torch.tensor([
            [*bboxes[0]],
            [*bboxes[1]],
            [*bboxes[-1]]
        ]).numpy()
        final_bboxes = non_max_suppression(bboxes, per_class=0)
        final_bboxes = final_bboxes.numpy()

        np.testing.assert_array_almost_equal(
            target,
            final_bboxes
        )
