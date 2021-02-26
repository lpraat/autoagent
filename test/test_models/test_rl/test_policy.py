import unittest
import numpy as np
import torch

from autoagent.models.rl.net import create_dense_net
from autoagent.models.rl.policy import CategoricalPolicy, GaussianPolicy, SquashedGaussianPolicy


class TestPolicy(unittest.TestCase):

    def test_categorical_policy(self):
        state_sizes = [np.random.randint(1, 20) for _ in range(5)]
        action_sizes = [np.random.randint(1, 10) for _ in range(5)]

        for s_size, a_size in zip(state_sizes, action_sizes):
            d = create_dense_net(
                input_size=s_size,
                output_size=a_size
            )

            policy = CategoricalPolicy(d)
            for _ in range(5):
                s = torch.randn(size=(1, s_size))
                a, log_p, _= policy(s, get_log_p=True)
                np.testing.assert_almost_equal(
                    log_p.detach().numpy(),
                    policy.log_p(s, a).detach().numpy()
                )

    def test_gaussian_policy(self):
        state_sizes = [np.random.randint(1, 20) for _ in range(5)]
        action_sizes = [np.random.randint(1, 10) for _ in range(5)]

        for s_size, a_size in zip(state_sizes, action_sizes):
            d = create_dense_net(
                input_size=s_size,
                output_size=a_size
            )

            policy = GaussianPolicy(d)
            for _ in range(5):
                s = torch.randn(size=(1, s_size))
                a, log_p, _ = policy(s, get_log_p=True)
                np.testing.assert_almost_equal(
                    log_p.detach().numpy(),
                    policy.log_p(s, a).detach().numpy()
                )

    def test_squashed_gaussian_policy(self):
        state_sizes = [np.random.randint(1, 20) for _ in range(5)]
        action_sizes = [np.random.randint(1, 10) for _ in range(5)]

        for s_size, a_size in zip(state_sizes, action_sizes):
            d = create_dense_net(
                input_size=s_size,
                output_size=a_size*2
            )

            policy = SquashedGaussianPolicy(d, action_limit=np.random.randint(1, 10, a_size))
            for _ in range(5):
                s = torch.randn(size=(1, s_size))
                a, log_p, _= policy(s, get_log_p=True)
                np.testing.assert_almost_equal(
                    log_p.detach().numpy(),
                    policy.log_p(s, a).detach().numpy(),
                    decimal=5
                )