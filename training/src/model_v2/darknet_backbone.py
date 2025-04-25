import torch
from torch import nn

from training.src.model_v2 import env


def conv_block(in_channels, out_channels, kernel_size=3, padding="same"):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )


class YoloV2(nn.Module):
    def __init__(self):
        super(YoloV2, self).__init__()
        self.stage1 = nn.Sequential(
            conv_block(3, 32),
            nn.MaxPool2d(2, 2),
            conv_block(32, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            conv_block(128, 64, 1),
            conv_block(64, 128, 1),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256),
            conv_block(256, 128, 1),
            conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512),
            conv_block(512, 256, 1),
            conv_block(256, 512),
            conv_block(512, 256, 1),
            conv_block(256, 512),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_block(512, 1024),
            conv_block(1024, 512, 1),
            conv_block(512, 1024),
            conv_block(1024, 512, 1),
            conv_block(512, 1024),
        )
        self.fcn = nn.Sequential(
            conv_block(1024 * 3, 1024),
            conv_block(1024, 1024),
            conv_block(1024, 1024),
        )
        self.output_layer = nn.Conv2d(
            in_channels=1024,
            out_channels=env.NO_OF_ANCHOR_BOX * (5 + env.NO_OF_CLASS),
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)

        # Skip connection from stage 1: slice into 4 parts and concatenate
        _, _, height, width = x1.size()
        part1 = x1[:, :, : height // 2, : width // 2]
        part2 = x1[:, :, : height // 2, width // 2:]
        part3 = x1[:, :, height // 2:, : width // 2]
        part4 = x1[:, :, height // 2:, width // 2:]
        residual = torch.cat((part1, part2, part3, part4), dim=1)

        # Concatenate residual with x2
        x_concat = torch.cat((x2, residual), dim=1)

        # Pass through FCN layers
        x3 = self.fcn(x_concat)

        # Pass through classifier
        out = self.output_layer(x3)

        # Original output shape: B, NO_ANCHORS * (5 + NO_CLASSES), 13, 13
        # Target shape: B, 13, 13, NO_ANCHORS, 5 + NO_CLASSES
        # Output has to be reshaped to match target shape
        new_out = out.permute(0, 2, 3, 1).contiguous()
        return new_out.view(
            new_out.size(0), new_out.size(1), new_out.size(2), env.NO_OF_ANCHOR_BOX, 5 + env.NO_OF_CLASS
        )
