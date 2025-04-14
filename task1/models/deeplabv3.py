import torch
from torch import nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_skip = self.conv(x)  # features before pooling
        x_pool = self.pool(x_skip)
        return x_pool, x_skip

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.out_conv(x)
        return x

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()
        # Encoder part using the EncoderBlock
        self.enc1 = EncoderBlock(3, 16)    # output shape is 1/2 of input size
        self.enc2 = EncoderBlock(16, 32)   # output shape is 1/4 of input size
        self.enc3 = EncoderBlock(32, 64)   # output shape is 1/8 of input size
        self.enc4 = EncoderBlock(64, 128)  # output shape is 1/16 of input size

        # ASPP module on the last encoder output
        self.aspp = ASPP(128, 64)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x, _ = self.enc1(x)  # feature map size: 1/2
        x, _ = self.enc2(x)  # feature map size: 1/4
        x, _ = self.enc3(x)  # feature map size: 1/8
        x, _ = self.enc4(x)  # feature map size: 1/16

        x = self.aspp(x)     # multi-scale feature extraction

        x = self.classifier(x)  # get segmentation logits

        # Upsample to the original input size
        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        x = torch.sigmoid(x)  # for binary segmentation output
        return x
