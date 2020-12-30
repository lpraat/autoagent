import math
import torch
import torch.nn as nn

from typing import List
from autoagent.models.yolo.layers import (
    Focus, Conv, CSPResidualBlock, SPP, CSPBottleneckBlock
)


class YoloBackbone(nn.Module):
    def __init__(self, mult_c, mult_d, act, conv):
        super().__init__()

        def get_c(c):
            return math.ceil(c*mult_c / 8) * 8

        def get_d(d):
            return max(round(d*mult_d), 1)

        # Init
        self.f1 = Focus(3, get_c(64), kernel_size=3, activation=act)

        # Down + cspblocks
        self.c1 = Conv(get_c(64), get_c(128), 3, stride=2, activation=act)
        self.cspres1 = CSPResidualBlock(get_c(128), get_c(128), get_d(3), activation=act, conv=conv)
        self.c2 = Conv(get_c(128), get_c(256), 3, 2, activation=act)
        self.cspres2 = CSPResidualBlock(get_c(256), get_c(256), get_d(9), activation=act, conv=conv)
        self.c3 = Conv(get_c(256), get_c(512), 3, 2, activation=act)
        self.cspres3 = CSPResidualBlock(get_c(512), get_c(512), get_d(9), activation=act, conv=conv)
        self.c4 = Conv(get_c(512), get_c(1024), 3, 2, activation=act)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.f1(x)
        x = self.cspres1(self.c1(x))
        x_s = self.cspres2(self.c2(x))
        x_m = self.cspres3(self.c3(x_s))
        x_l = self.c4(x_m)
        return [x_l, x_m, x_s]


class YoloHead(nn.Module):
    def __init__(self, num_classes, mult_c, mult_d, act, conv):
        super().__init__()

        def get_c(c):
            return math.ceil(c*mult_c / 8) * 8

        def get_d(d):
            return max(round(d*mult_d), 1)

        # SPP
        self.spp = SPP(get_c(1024), get_c(1024), activation=act, conv=conv)

        # Top-down (PAN (a))
        self.cspbn1 = CSPBottleneckBlock(get_c(1024), get_c(1024), get_d(3), activation=act, conv=conv)
        self.c1 = conv(get_c(1024), get_c(512), 1, activation=act)
        self.u1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.cspbn2 = CSPBottleneckBlock(get_c(1024), get_c(512), get_d(3), activation=act, conv=conv)
        self.c2 = conv(get_c(512), get_c(256), 1, activation=act)
        self.u2 = nn.Upsample(scale_factor=2, mode='nearest')

        # Bottom-up (PAN (b))
        self.cspbn3 = CSPBottleneckBlock(get_c(512), get_c(256), get_d(3), activation=act, conv=conv)
        self.c3 = conv(get_c(256), get_c(256), 3, stride=2, activation=act)

        self.cspbn4 = CSPBottleneckBlock(get_c(512), get_c(512), get_d(3), activation=act, conv=conv)
        self.c4 = conv(get_c(512), get_c(512), 3, stride=2, activation=act)

        self.cspbn5 = CSPBottleneckBlock(get_c(1024), get_c(1024), get_d(3), activation=act, conv=conv)

        self.fs = nn.Conv2d(get_c(256), 3*(5+num_classes), 1)
        self.fm = nn.Conv2d(get_c(512), 3*(5+num_classes), 1)
        self.fl = nn.Conv2d(get_c(1024), 3*(5+num_classes), 1)

    def init_biases(self):
        for l in [self.fl, self.fm, self.fs]:
            x = l.bias
            x = x.view(3, -1)
            x[:, 4:] = -5
            l.bias = nn.Parameter(x.view(-1), requires_grad=True)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x_l, x_m, x_s = x[0], x[1], x[2]

        # SPP
        x = self.spp(x_l)

        # Top-down
        x_10 = self.c1(self.cspbn1(x))
        x = self.u1(x_10)
        x = torch.cat([x, x_m], dim=1)

        x_14 = self.c2(self.cspbn2(x))
        x = self.u2(x_14)
        x = torch.cat([x, x_s], dim=1)

        # Bottom-up
        x = self.cspbn3(x)
        o_s = self.fs(x)
        x = self.c3(x)
        x = torch.cat([x, x_14], dim=1)

        x = self.cspbn4(x)
        o_m = self.fm(x)
        x = self.c4(x)
        x = torch.cat([x, x_10], dim=1)

        x = self.cspbn5(x)
        o_l = self.fl(x)

        return o_l, o_m, o_s


class YoloModel(nn.Module):
    """
    CSP + SPP + PAN w/ multipliers.
    """
    def __init__(self, num_classes, mult_c, mult_d, activation=nn.LeakyReLU, conv=Conv):
        super().__init__()
        self.backbone = YoloBackbone(mult_c, mult_d, activation, conv)
        self.head = YoloHead(num_classes, mult_c, mult_d, activation, conv)

    def forward(self, x):
        return self.head(self.backbone(x))
