import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim + self.class_dim,
                               out_channels=1024,
                               kernel_size=4,
                               stride=1,
                               bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=3,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x, c):
        concat = torch.cat((x, c), dim=1)  # Concatenate noise and class vector.
        concat = concat.unsqueeze(2).unsqueeze(3)  # Reshape the latent vector into a feature map.

        return self.main(concat)


class Discriminator(nn.Module):
    def __init__(self, hair_classes, eyes_classes):
        super(Discriminator, self).__init__()

        self.hair_classes = hair_classes
        self.eyes_classes = eyes_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=1,
                      kernel_size=4,
                      stride=1),
            nn.Sigmoid()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=512,
                      kernel_size=4,
                      stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(512, self.hair_classes + self.eyes_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_layers(x)
        discrim_output = self.discriminator_layer(features).view(-1)  # Single-value scalar
        flatten = self.bottleneck(features).squeeze()
        class_output = self.classifier_layer(flatten)  # Outputs probability for each class label

        return discrim_output, class_output
