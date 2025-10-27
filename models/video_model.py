import torch
import torch.nn as nn

class CubePad(nn.Module):
    """
    Input : [6N, C, H, W]
    Output: [6N, C, H+2p, W+2p] when lrtd_pad=p
    """
    def __init__(self, lrtd_pad=1, use_gpu=True):
        super().__init__()
        from cube_pad import CubePadding
        self.lrtd_pad = lrtd_pad
        self.cp = CubePadding(lrtd_pad, use_gpu)

    def forward(self, x):
        b6, c, h, w = x.shape
        assert b6 % 6 == 0, f"CubePad: batch {b6} not divisible by 6"
        N = b6 // 6
        x = x.view(N, 6, c, h, w)
        x = self.cp(x)
        return x.view(N * 6, c, h + 2*self.lrtd_pad, w + 2*self.lrtd_pad)

# 3x3 conv that uses cube padding
class CPConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.pad = CubePad(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class CPBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = CPConv3x3(in_ch,  out_ch)
        self.c2 = CPConv3x3(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))

class CubeUNet(nn.Module):
    """
    Input per face: 4ch (RGB3 + motion1).
    Forward expects x shaped [B*6, 4, S, S].
    Returns [B*6, 1, S, S].
    """
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.enc1 = CPBlock(in_ch,     base)      # S
        self.pool1 = nn.MaxPool2d(2)              # S/2
        self.enc2 = CPBlock(base,      base*2)    # S/2
        self.pool2 = nn.MaxPool2d(2)              # S/4
        self.enc3 = CPBlock(base*2,    base*4)    # S/4
        self.pool3 = nn.MaxPool2d(2)              # S/8
        self.enc4 = CPBlock(base*4,    base*8)    # S/8

        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # S/4
        self.dec1 = CPBlock(base*8 + base*4, base*4)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # S/2
        self.dec2 = CPBlock(base*4 + base*2, base*2)
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # S
        self.dec3 = CPBlock(base*2 + base,   base)

        self.head = nn.Conv2d(base, 1, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d1 = self.up1(e4)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        return self.out_act(self.head(d3))
