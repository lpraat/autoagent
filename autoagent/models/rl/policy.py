import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class BasicPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, get_log_p=True, deterministic=False):
        raise NotImplementedError()

    def log_p(self, s, a):
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
        dist = Categorical(probs=probs)
        if deterministic:
            a = torch.argmax(probs, dim=1)
        else:
            a = dist.sample()
        log_p = dist.log_prob(a) if get_log_p else None
        return a, log_p, dist

    def log_p(self, s, a):
        probs = nn.functional.softmax(self.net(s), dim=1)
        return Categorical(probs=probs).log_prob(a.squeeze())

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

    def forward(self, s, get_log_p=True, deterministic=False):
        mean = self.mean_net(s)
        dist = Normal(mean, torch.exp(self.log_std))
        a = mean if deterministic else dist.sample()
        log_p = dist.log_prob(a).sum(dim=1) if get_log_p else None
        return a, log_p, dist

    def log_p(self, s, a):
        return Normal(self.mean_net(s), torch.exp(self.log_std)).log_prob(a).sum(dim=1)

    def predict(self, s, deterministic=False):
        return self(s, get_log_p=False, deterministic=deterministic)[0]


class SquashedGaussianPolicy(BasicPolicy):
    """
    Squashed (using tanh) Gaussian Policy
    with state-dependent diagonal covariance matrix
    """
    def __init__(self, net, action_limit, log_std_bounds=(-20, 2)):
        super().__init__()
        self.net = net
        self.log_of_two_pi = torch.tensor(np.log(2*np.pi), dtype=torch.float32)
        self.log_of_two = torch.tensor(np.log(2), dtype=torch.float32)
        assert (action_limit > 0).all()
        self.action_limit = torch.tensor(action_limit, dtype=torch.float32)
        self.log_std_bounds = log_std_bounds

    def _compute_log_p(self, a, mean, log_std):
        # Compute log probabilities
        # (multivariate gaussian term + change of variable correction)
        # Formula can be obtained by using:
        # - integration by substition + derivative of inverse function
        # - this identity: log(1 - tanh(x)**2) == 2*(log(2) - x - softplus(-2*x)))
        #   or equiv.      log(1 - tanh(x)**2) == 2*(log(2) + x - softplus(2*x))
        #  for numerical stability
        log_p = torch.sum(
            -0.5 * (
                self.log_of_two_pi
                + 2*log_std
                + ((a - mean) / torch.exp(log_std)) ** 2
            ), dim=1)
        log_p -= (
            2 * (
                self.log_of_two
                - a
                - torch.nn.functional.softplus(-2*a)
            ) + torch.log(self.action_limit)).sum(dim=1)
        return log_p

    def forward(self, s, get_log_p=True, deterministic=False):
        net_output = self.net(s)
        mean, log_std = torch.split(net_output,
                                    int(net_output.shape[1]/2), dim=1)
        log_std = torch.clamp(log_std, *self.log_std_bounds)

        if deterministic:
            a = mean
        else:
            a = mean + torch.randn(mean.size()) * torch.exp(log_std)

        squashed_a = torch.tanh(a) * self.action_limit

        log_p = self._compute_log_p(a, mean, log_std) if get_log_p else None
        return squashed_a, log_p, (mean, log_std)

    def log_p(self, s, a):
        a = torch.clamp(a / self.action_limit, -1+1e-7, 1-1e-7)
        a = torch.atanh(a)
        net_output = self.net(s)
        mean, log_std = torch.split(net_output,
                                    int(net_output.shape[1]/2), dim=1)
        return self._compute_log_p(a, mean, log_std)

    @torch.no_grad()
    def predict(self, s, deterministic=False):
        return self(s, get_log_p=False, deterministic=deterministic)[0]
