import torch
import torch.nn as nn


class Focus(nn.Module):
    """
    https://github.com/ultralytics/yolov5/blob/master/models/common.py
    """
    # Focus wh information into c-space
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, activation=nn.LeakyReLU):
        super(Focus, self).__init__()
        self.conv = Conv(in_c * 4, out_c, kernel_size, stride, activation)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, activation=nn.LeakyReLU):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

        if activation is nn.LeakyReLU:
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = activation()

        if hasattr(self.activation, 'inplace'):
            self.activation.inplace = True

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_c, hid_c, out_c, activation, conv):
        super().__init__()
        self.c1 = conv(in_c, hid_c, 1, activation=activation)
        self.c2 = conv(hid_c, out_c, 3, activation=activation)

    def forward(self, x):
        return self.c2(self.c1(x))


class BottleneckStack(nn.Module):
    def __init__(self, in_c, hid_c, out_c, stack_size, activation, conv):
        super().__init__()
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(in_c, hid_c, out_c, activation, conv) for _ in range(stack_size)
        ])

    def forward(self, x):
        return self.bottlenecks(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, hid_c, out_c, activation, conv):
        super().__init__()
        self.bottleneck = Bottleneck(in_c, hid_c, out_c, activation, conv)

    def forward(self, x):
        return x + self.bottleneck(x)


class ResidualStack(nn.Module):
    def __init__(self, in_c, hid_c, out_c, stack_size, activation, conv):
        super().__init__()
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(in_c, hid_c, out_c, activation, conv) for _ in range(stack_size)
        ])

    def forward(self, x):
        return self.res_blocks(x)


class CSPResidualBlock(nn.Module):
    """
    CSP for Residual.
    https://arxiv.org/pdf/1911.11929.pdf, Figure 2.
    https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    def __init__(self, in_c, out_c, size, activation, conv, narrow=True):
        super().__init__()
        hid_c = out_c//2 if narrow else out_c
        self.right = conv(in_c, hid_c, 1, activation=activation)

        self.left = nn.Conv2d(in_c, hid_c, 1, bias=False)
        self.c1 = nn.Conv2d(hid_c, hid_c, 1, bias=False)
        self.fc = conv(hid_c*2, out_c, 1, activation=activation)

        self.bn1 = nn.BatchNorm2d(out_c)
        self.a1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc = conv(hid_c*2, out_c, 1, activation=activation)
        self.rs = ResidualStack(hid_c, hid_c, hid_c, size, activation=activation, conv=conv)

    def forward(self, x):
        left = self.c1(self.rs(self.right(x)))
        right = self.left(x)
        return self.fc(self.a1(self.bn1(torch.cat([left, right], dim=1))))


class CSPBottleneckBlock(nn.Module):
    """
    CSP for Bottleneck.
    https://github.com/ultralytics/yolov5/blob/7220cee1d1dc1f14003dbf8d633bbb76c547000c/models/common.py#L49
    """
    def __init__(self, in_c, out_c, size, activation, conv, narrow=True):
        super().__init__()
        hid_c = out_c//2 if narrow else out_c
        self.right = conv(in_c, hid_c, 1, activation=activation)

        self.left = nn.Conv2d(in_c, hid_c, 1, bias=False)
        self.c1 = nn.Conv2d(hid_c, hid_c, 1, bias=False)
        self.fc = conv(hid_c*2, out_c, 1, activation=activation)

        self.bn1 = nn.BatchNorm2d(out_c)
        self.a1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc = conv(hid_c*2, out_c, 1, activation=activation)
        self.rs = BottleneckStack(hid_c, hid_c, hid_c, size, activation=activation, conv=conv)

    def forward(self, x):
        left = self.c1(self.rs(self.right(x)))
        right = self.left(x)
        return self.fc(self.a1(self.bn1(torch.cat([left, right], dim=1))))


class SPP(nn.Module):
    """Spatial Pyramid Pooling, https://arxiv.org/ftp/arxiv/papers/1903/1903.08589.pdf"""
    def __init__(self, c_in, c_out, activation, conv):
        super().__init__()
        self.c1 = conv(c_in, c_in//2, 1, activation=activation)
        self.c2 = conv(c_in//2 * 4, c_out, 1, activation=activation)

        self.m1 = nn.MaxPool2d(5, stride=1, padding=5//2)
        self.m2 = nn.MaxPool2d(9, stride=1, padding=9//2)
        self.m3 = nn.MaxPool2d(13, stride=1, padding=13//2)

    def forward(self, x):
        x = self.c1(x)
        return self.c2(torch.cat([x, self.m1(x), self.m2(x), self.m3(x)], dim=1))
