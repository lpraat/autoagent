import torch
import torch.nn as nn


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