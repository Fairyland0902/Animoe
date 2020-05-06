import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.leaky_relu_2 = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        output = self.conv_1(x)
        output = self.leaky_relu_1(output)
        output = self.conv_2(output)
        output += residual
        output = self.leaky_relu_2(output)
        return output


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, conv_stride=2, conv_kernel_size=4):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=padding, bias=bias)
        self.leaky_relu = nn.LeakyReLU()
        self.residual_block_1 = ResidualBlock(out_channels, out_channels, kernel_size, stride)
        self.residual_block_2 = ResidualBlock(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        output = self.conv(x)
        output = self.leaky_relu(output)
        output = self.residual_block_1(output)
        output = self.residual_block_2(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, tag=34):
        super(Discriminator, self).__init__()
        self.reduce_block_1 = Block(3, 32, conv_kernel_size=4)
        self.reduce_block_2 = Block(32, 64, conv_kernel_size=4)
        self.reduce_block_3 = Block(64, 128, conv_kernel_size=4)
        self.reduce_block_4 = Block(128, 256, conv_kernel_size=3)
        self.reduce_block_5 = Block(256, 512, conv_kernel_size=3)
        self.conv = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dense_label = nn.Linear(2 * 2 * 1024, 1)
        self.dense_tag = nn.Linear(2 * 2 * 1024, tag)

    def forward(self, x):
        output = self.reduce_block_1(x)
        output = self.reduce_block_2(output)
        output = self.reduce_block_3(output)
        output = self.reduce_block_4(output)
        output = self.reduce_block_5(output)
        output = self.conv(output)
        output = self.leaky_relu(output)
        output = output.view(output.size(0), -1)
        return self.dense_label(output), self.dense_tag(output)
