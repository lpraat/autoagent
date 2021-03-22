import torch
import torch.nn as nn
import numpy as np


def create_dense_net(
        input_size,
        output_size,
        hidden_sizes=[64, 64],
        hidden_activation=nn.ReLU,
        output_activation=nn.Identity,
        weights_init_=torch.nn.init.orthogonal_
    ):
    layers = []

    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.extend([
                nn.Linear(input_size, hidden_sizes[i]),
                hidden_activation()
            ])
        else:
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                hidden_activation()
            ])

    if output_activation is not nn.Identity:
        layers.extend([
            nn.Linear(hidden_sizes[i], output_size), output_activation()
        ])
    else:
        layers.append(
            nn.Linear(hidden_sizes[i], output_size)
        )
    net = nn.Sequential(*layers)

    for module in net:
        if isinstance(module, nn.Linear):
            weights_init_(module.weight)

    return net


def create_dense_net_from_layers(
        *layers,
        weights_init_=torch.nn.init.orthogonal_
    ):
    net = nn.Sequential(*layers)
    for module in net:
        if isinstance(module, nn.Linear):
            weights_init_(module.weight)
    return net


def initialized_conv2d(*args, weights_init_=torch.nn.init.orthogonal_, **kwargs):
    conv = nn.Conv2d(*args, **kwargs)
    weights_init_(conv.weight)
    return conv


def initialized_linear(*args, weights_init_=torch.nn.init.orthogonal_, **kwargs):
    l = nn.Linear(*args, **kwargs)
    weights_init_(l.weight)
    return l


def create_impala_cnn(img_size, block_chs=(16, 32, 32), out_dim=256,
                      weights_init_=torch.nn.init.orthogonal_):
    """
    https://arxiv.org/pdf/1802.01561.pdf, Page 5, Figure 3.
    """
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.res = nn.Sequential(
                nn.ReLU(),
                initialized_conv2d(ch, ch, kernel_size=3, stride=1, padding=1,
                                   weights_init_=weights_init_),
                nn.ReLU(),
                initialized_conv2d(ch, ch, kernel_size=3, stride=1, padding=1,
                                   weights_init_=weights_init_)
            )

        def forward(self, x):
            return x + self.res(x)

    class ImpalaBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.seq = nn.Sequential(
                initialized_conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                                   weights_init_=weights_init_),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(out_ch),
                ResBlock(out_ch)
            )

        def forward(self, x):
            return self.seq(x)

    class ImpalaCNN(nn.Module):
        def __init__(self, img_size, out_dim=256, block_chs=(16, 32, 32)):
            super().__init__()

            # (h, w, ch)
            self.img_size = img_size

            # Compute impala blocks output dimension
            blocks_out_dim = np.array(img_size)
            for _ in range(len(block_chs)):
                blocks_out_dim = (blocks_out_dim+1) // 2
            blocks_out_dim[2] = block_chs[-1]
            self.blocks_out_dim = np.prod(blocks_out_dim)

            self.impala_blocks = [
                ImpalaBlock(img_size[2], block_chs[0]),
                *[ImpalaBlock(block_chs[i-1], block_chs[i]) for i in range(1, len(block_chs))]
            ]
            self.seq = nn.Sequential(
                *self.impala_blocks,
                nn.ReLU(),
                nn.Flatten(),
                initialized_linear(self.blocks_out_dim, out_dim,
                                   weights_init_=weights_init_),
                nn.ReLU()
            )

        def forward(self, x):
            x /= 255.0
            x = x.view(-1, *self.img_size).permute(0, 3, 1, 2)
            return self.seq(x)

    impala_cnn = ImpalaCNN(img_size, out_dim, block_chs)
    return impala_cnn
