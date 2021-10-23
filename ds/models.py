import torch
from torch.nn import functional as F
from torch import nn


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
        nn.Dropout(dropout_p),
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
        self.conv_1 = conv_block(12, 16, 5)
        self.conv_2 = conv_block(16, 32, 5)
        self.conv_3 = conv_block(32, 64, 3, 0.2)
        self.conv_4 = conv_block(64, 256, 3, 0.2)

        self.ln_1 = lin_block(125*64, 64)
        self.ln_2 = lin_block(64, 64)
        self.ln_3 = lin_block(64, 5)
        self.ln_4 = lin_block(200, 100)
        self.ln_5 = lin_block(100, 50)
        self.ln_6 = lin_block(50, 10)
        self.ln_7 = lin_block(10, 5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = self.conv_1(wave)
        wave = self.conv_2(wave)
        wave = self.conv_3(wave)

        wave = torch.flatten(wave, 1)

        wave = F.relu(self.ln_1(wave))
        wave = F.relu(self.ln_2(wave))
        wave = self.ln_3(wave)

        return torch.sigmoid(wave)
