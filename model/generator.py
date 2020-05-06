import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        output = self.conv_1(x)
        output = self.bn_1(output)
        output = self.relu(output)
        output = self.conv_2(output)
        output = self.bn_2(output)
        output += residual
        return output


class SubpixelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False, upscale_factor=2):
        super(SubpixelBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        output = self.conv(x)
        output = self.pixel_shuffle(output)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Generator(nn.Module):
    def __init__(self, tag=34):
        super(Generator, self).__init__()
        in_channels = 128 + tag
        self.dense = nn.Linear(in_channels, 64 * 16 * 16)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.PReLU()
        self.residual_layer = self.ResidualLayer(16)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.PReLU()
        self.subpixel_layer = self.SubpixelLayer(3)
        self.conv = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.dense(x)
        output = output.view(-1, 64, 16, 16)
        output = self.bn_1(output)
        output = self.relu_1(output)
        residual = output
        output = self.residual_layer(output)
        output = self.bn_2(output)
        output = self.relu_2(output)
        output += residual
        output = self.subpixel_layer(output)
        output = self.conv(output)
        output = self.tanh(output)
        return output

    def ResidualLayer(self, block_size=16):
        layers = []
        for _ in range(block_size):
            layers.append(ResidualBlock(64, 64, 3, 1))
        return nn.Sequential(*layers)

    def SubpixelLayer(self, block_size=3):
        layers = []
        for _ in range(block_size):
            layers.append(SubpixelBlock(64, 256, 3, 1))
        return nn.Sequential(*layers)
