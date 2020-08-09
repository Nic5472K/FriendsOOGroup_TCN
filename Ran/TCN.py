# https://arxiv.org/abs/1803.01271

import torch.nn as nn


class ResidualBlock(nn.Module):
    """One residual block of codebase.
    """

    def __init__(self, dilation_factor, in_channels, intermediate_channels, out_channels, kernel_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.dilation_factor = dilation_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.padding_size = dilation_factor * (kernel_size - 1)

        self.conv1d1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=kernel_size,
                stride=1, padding=self.padding_size,
                dilation=dilation_factor,
                groups=1, bias=True, padding_mode="zeros"
            ), name="weight"
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv1d2 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1, padding=self.padding_size,
                dilation=dilation_factor,
                groups=1, bias=True, padding_mode="zeros"
            ), name="weight"
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.conv1d1by1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (N, C_in, L_in)
        Returns:
            output: (N, C_out, L_out)
        """
        layer1 = self.dropout1(self.relu1(self.conv1d1(x)[:, :, :-self.padding_size].contiguous()))
        layer2 = self.dropout2(self.relu2(self.conv1d2(layer1)[:, :, :-self.padding_size].contiguous()))

        # residual connection and (optional) 1x1 conv
        if self.in_channels != self.out_channels:
            return self.conv1d1by1(x) + layer2
        else:
            return x + layer2


class TCN(nn.Module):
    def __init__(self, channels_list, intermediate_channels, kernel_size, dropout=0.2):
        """
        Args:
            channels_list: [int], including input channels, every middle layer channels, and output channel
            intermediate_channels: int, channels in residual block
            kernel_size: int
            dropout: float
        """
        super(TCN, self).__init__()
        n_layers = len(channels_list)
        layers = []
        for i in range(n_layers - 1):
            dilation_factor = 2 ** i
            layers.append(
                ResidualBlock(
                    dilation_factor=dilation_factor,
                    in_channels=channels_list[i],
                    intermediate_channels=intermediate_channels,
                    out_channels=channels_list[i + 1],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input: (N, C_in, L)
        Returns:
            (N, C_out, L)
        """
        return self.network(input)
