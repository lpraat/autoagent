import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class BasicPolicy(nn.Module):
    def forward(self, s, get_log_p=True, deterministic=False):
        raise NotImplementedError()

    def predict(self, s, deterministic=False):
        raise NotImplementedError()


class CategoricalPolicy(BasicPolicy):
    """
    Categorical policy
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, s, get_log_p=True, deterministic=False):
        probs = nn.functional.softmax(self.net(s), dim=1)
        cat = Categorical(probs=probs)
        if deterministic:
            a = torch.argmax(probs)
        else:
            a = cat.sample()
        log_p = cat.log_prob(a) if get_log_p else None
        return a, log_p

    @torch.no_grad()
    def predict(self, s, deterministic=False):
        return self(s, get_log_p=False, deterministic=deterministic)[0]


class GaussianPolicy(BasicPolicy):
    """
    Gaussian policy with state-independent diagonal covariance matrix
    """

    def __init__(self, mean_net, log_std_init=-0.5):
        super().__init__()
        self.mean_net = mean_net
        self.log_std = nn.Parameter(
            log_std_init * torch.ones(
                self.mean_net[-1].out_features, dtype=torch.float32
            )
        )

    def forward(self, x, get_log_p=True, deterministic=False):
        mean = self.mean_net(x)
        n = Normal(mean, torch.exp(self.log_std))
        a = mean if deterministic else n.sample()
        log_p = n.log_prob(a).sum(dim=1) if get_log_p else None
        return a, log_p

    def predict(self, s, deterministic=False):
        return self(s, get_log_p=False, deterministic=deterministic)[0]


class SquashedGaussianPolicy(BasicPolicy):
    """
    Squashed (using tanh) Gaussian Policy
    with state-dependent diagonal covariance matrix
    """
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, net, action_limit):
        super().__init__()
        self.net = net

        self.log_of_two_pi = (
            'log_of_two_pi',
            torch.tensor(np.log(np.pi), dtype=torch.float32)
        )
        self.register_buffer(
            'log_of_two',
            torch.tensor(np.log(2), dtype=torch.float32)
        )
        self.register_buffer(
            'action_limit',
            torch.tensor(action_limit, dtype=torch.float32)
        )

    def forward(self, s, get_log_p=False, deterministic=False):
        net_output = self.net(s)
        mean, log_std = torch.split(net_output,
                                    int(net_output.shape[1]/2), dim=1)
        log_std = torch.clamp(log_std,
                              SquashedGaussianPolicy.LOG_STD_MIN,
                              SquashedGaussianPolicy.LOG_STD_MAX)

        if deterministic:
            a = mean
        else:
            a = mean + torch.randn(mean.size()) * torch.exp(log_std)

        squashed_a = torch.tanh(a) * self.action_limit

        log_p = None
        if get_log_p:
            # Compute log probabilities
            # (multivariate gaussian term + change of variable correction)
            # Formula can be obtained by using:
            # - integration by substition + derivative of inverse function
            # - this identity: log(1 - tanh(x)**2) == 2*(log(2) - x - softplus(-2*x)))
            #   or equiv.      log(1 - tanh(x)**2) == 2*(log(2) + x - softplus(2*x))
            #  for numerical stability
            log_p = torch.sum(
                -0.5 * (
                    self.log_of_pi
                    + 2*log_std
                    + ((a - mean) / torch.exp(log_std)) ** 2
                ), dim=1)
            log_p -= (
                2 * (
                    self.log_of_two
                    - a
                    - torch.nn.functional.softplus(-2*a)
                ) + torch.log(torch.abs(self.action_limit))).sum(dim=1)

        return squashed_a, log_p

    @torch.no_grad()
    def predict(self, s, deterministic=False):
        return self(s, get_log_p=False, deterministic=deterministic)[0]
