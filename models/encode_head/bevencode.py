import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1 shape: torch.Size([1, 256, 13, 13])
        # x2 shape: torch.Size([1, 64, 50, 50])
        x1 = self.up(x1)    # torch.Size([1, 256, 52, 52])
        x1 = torch.cat([x2, x1], dim=1) # torch.Size([1, 192, 50, 50])
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        # self.layer3 = trunk.layer3

        # changed from 4 to 2 bcoz we are removing the layer 3 
        # and 256 to 128 in (64+128)
        self.up1 = Up(64+128, 256, scale_factor=2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
        # Additional upsampling layer to scale from 100x100 to 200x200
        # self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # input shape: torch.Size([1, 64, 100, 100])
        x = self.conv1(x)   # torch.Size([1, 64, 50, 50])
        x = self.bn1(x)     # torch.Size([1, 64, 50, 50])
        x = self.relu(x)    # torch.Size([1, 64, 50, 50])

        x1 = self.layer1(x) # torch.Size([1, 64, 50, 50])
        x = self.layer2(x1) # torch.Size([1, 128, 25, 25])
        # we are having 100 instead of 200 as done by LSS
        # x = self.layer3(x)  # torch.Size([1, 256, 13, 13])

        x = self.up1(x, x1)     # torch.Size([1, 256, 50, 50])
        x = self.up2(x)         # torch.Size([1, 1, 100, 100])
        # x = self.up3(x)         # torch.Size([1, 1, 200, 200])

        return x
