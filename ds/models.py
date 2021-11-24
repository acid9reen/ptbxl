from functools import partial

import torch
from torch.nn import functional as F
from torch import nn


def activation_func(activation):
    return nn.ModuleDict(
        {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01, inplace=True),
            "selu": nn.SELU(inplace=True),
            "none": nn.Identity(),
        }
    )[activation]


class Conv1dAuto(nn.Conv1d):
    """Auto padding tweak to pytorch Conv1d"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)


conv3 = partial(Conv1dAuto, kernel_size=(3,), bias=False)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "relu"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.activate = activation_func(activation)
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        x = self.blocks(x)
        x += residual
        x = self.activate(x)

        return x

    @property
    def should_apply_shortcut(self) -> bool:
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 1,
        downsampling: int = 1,
        conv: nn.Conv1d = conv3,
        *args,
        **kwargs
    ) -> None:
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv

        self.shortcut = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.expanded_channels,
                kernel_size=(1,),
                stride=(self.downsampling,),
                bias=False,
            ),
            nn.BatchNorm1d(self.expanded_channels)
            if self.should_apply_shortcut
            else None,
        )

    @property
    def expanded_channels(self) -> int:
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self) -> bool:
        return self.in_channels != self.expanded_channels


def conv_bn(
    in_channels: int, out_channels: int, conv: nn.Conv1d, *args, **kwargs
) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm1d(out_channels)
    )


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                self.in_channels,
                self.out_channels,
                conv=self.conv,
                bias=False,
                stride=self.downsampling,
            ),
            activation_func(self.activation),
            nn.Dropout(0.2),
            conv_bn(
                self.out_channels, self.expanded_channels, conv=self.conv, bias=False
            ),
        )


class ResNetLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block=ResNetBasicBlock,
        n: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(
                in_channels, out_channels, *args, **kwargs, downsampling=downsampling
            ),
            *[
                block(
                    out_channels * block.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs
                )
                for __ in range(n - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)

        return x


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        blocks_sizes: tuple[int] = (64, 128, 256, 512),
        depths: tuple[int] = (1, 1, 1, 1),
        activation: str = "relu",
        block=ResNetBasicBlock,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv1d(
                in_channels,
                self.blocks_sizes[0],
                kernel_size=(7,),
                stride=(2,),
                padding=(3,),
                bias=False,
            ),
            nn.BatchNorm1d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_sizes = zip(blocks_sizes, blocks_sizes[1:])

        self.blocks = nn.ModuleList(
            [
                ResNetLayer(
                    blocks_sizes[0],
                    blocks_sizes[0],
                    n=depths[0],
                    activation=activation,
                    block=block,
                    *args,
                    **kwargs
                ),
                *[
                    ResNetLayer(
                        in_channels * block.expansion,
                        out_channels,
                        n=n,
                        activation=activation,
                        block=block,
                        *args,
                        **kwargs
                    )
                    for (in_channels, out_channels), n in zip(
                        self.in_out_block_sizes, depths[1:]
                    )
                ],
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)

        for block in self.blocks:
            x = block(x)

        return x


class ResnetDecoder(nn.Module):
    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d((1,))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.decoder(x)

        return torch.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def basic_res_net(
    in_channels: int, n_classes: int, block=ResNetBasicBlock, *args, **kwargs
) -> ResNet:

    return ResNet(
        in_channels, n_classes, block=block, depths=(2, 2, 2, 2), *args, **kwargs
    )


def conv_block(
        input_size: int,
        output_size: int,
        kernel_size: int,
        dropout_p: float=0.5) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv1d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        ),
        nn.BatchNorm1d(output_size),
        nn.ReLU(),
        #nn.Dropout(dropout_p),
        nn.MaxPool1d(2),
    )

    return block


def lin_block(
        input_size: int,
        output_size: int) -> nn.Sequential:
    block = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.BatchNorm1d(output_size),
    )

    return block


class ConvLinearBasicModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = conv_block(12, 16, 5, 0.5)
        self.conv_2 = conv_block(16, 32, 3, 0.5)
        self.conv_3 = conv_block(32, 64, 3, 0.5)
        self.conv_4 = conv_block(64, 256, 3, 0.5)

        self.ln_1 = lin_block(62*256, 128)
        self.ln_2 = lin_block(64, 64)
        self.ln_3 = lin_block(64, 5)
        self.ln_4 = lin_block(128, 64)
        self.ln_5 = lin_block(100, 50)
        self.ln_6 = lin_block(50, 10)
        self.ln_7 = lin_block(10, 5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = self.conv_1(wave)
        wave = self.conv_2(wave)
        wave = self.conv_3(wave)
        wave = self.conv_4(wave)

        wave = torch.flatten(wave, 1)

        wave = F.relu(self.ln_1(wave))
        wave = F.relu(self.ln_4(wave))
        wave = F.relu(self.ln_2(wave))
        wave = self.ln_3(wave)

        return torch.sigmoid(wave)
