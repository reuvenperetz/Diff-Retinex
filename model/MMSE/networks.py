import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        if norm:
            layers.insert(1, nn.BatchNorm2d(out_ch))
            layers.insert(-1, nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, norm=True):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, norm=norm)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, norm=norm)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base_ch * 2, base_ch * 4, norm=norm)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, norm=norm)

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def build_mmse_net(arch, in_ch, out_ch, base_ch=32, norm=True):
    arch = arch.lower()
    if arch == "unet":
        return SimpleUNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, norm=norm)
    if arch == "cnn":
        return SmallCNN(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
    raise ValueError(f"Unknown MMSE arch: {arch}")
