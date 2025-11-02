import torch
import torch.nn as nn
import torch.nn.functional as F

class CubePad(nn.Module):
    """
    Input : [6N, C, H, W]
    Output: [6N, C, H+2p, W+2p]
    """
    def __init__(self, lrtd_pad=1, use_gpu=True):
        super().__init__()
        from utils.cube_pad import CubePadding
        self.lrtd_pad = lrtd_pad
        self.cp = CubePadding(lrtd_pad, use_gpu)

    def forward(self, x):
        b6, c, h, w = x.shape
        assert b6 % 6 == 0, f"CubePad: batch {b6} not divisible by 6"
        N = b6 // 6
        x = x.view(N, 6, c, h, w)
        x = self.cp(x)
        return x.view(N * 6, c, h + 2*self.lrtd_pad, w + 2*self.lrtd_pad)

class CPConv3x3(nn.Module):
    """ cube-pad -> 3×3 conv -> BN -> ReLU """
    def __init__(self, in_ch, out_ch, stride=1, bias=False):
        super().__init__()
        self.pad = CubePad(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=0, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

# Cube Safe ResNet Encoder
class _CPBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = CPConv3x3(inplanes, planes, stride=stride)
        self.conv2 = CPConv3x3(planes,  planes, stride=1)
        self.down  = downsample
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class _ResEnc(nn.Module):
    """
    ResNet18 like, cube aware:
    - Stem: 7x7/2 with CubePad(3); NO maxpool
    - Layers: [2,2,2,2] with strides [1,2,2,2]
    - 4 input channels (RGB+motion)
    """
    def __init__(self, in_ch=4, pretrained=True):
        super().__init__()
        self.stem_pad = CubePad(3)
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.inplanes = 64
        self.layer1 = self._make_layer(64,  blocks=2, stride=1)  # -> S/2
        self.layer2 = self._make_layer(128, blocks=2, stride=2)  # -> S/4
        self.layer3 = self._make_layer(256, blocks=2, stride=2)  # -> S/8
        self.layer4 = self._make_layer(512, blocks=2, stride=2)  # -> S/16

        if pretrained:
            self._load_torchvision_weights()

    def _make_layer(self, planes, blocks, stride):
        down = None
        if stride != 1 or self.inplanes != planes:
            down = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_CPBasicBlock(self.inplanes, planes, stride, down)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_CPBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _load_torchvision_weights(self):
        from torchvision.models import resnet18, ResNet18_Weights
        ref = resnet18(weights=ResNet18_Weights.DEFAULT)

        # widen conv1 3->4 (copy mean RGB into motion)
        w = ref.conv1.weight
        new_w = torch.zeros(64, 4, 7, 7, dtype=w.dtype, device=w.device)
        new_w[:, :3] = w
        new_w[:, 3]  = w.mean(dim=1)
        self.conv1.weight.copy_(new_w)
        self.bn1.load_state_dict(ref.bn1.state_dict())

        def copy_block(dst, src):
            dst.conv1.conv.weight.copy_(src.conv1.weight)
            dst.conv1.bn.load_state_dict(src.bn1.state_dict())
            dst.conv2.conv.weight.copy_(src.conv2.weight)
            dst.conv2.bn.load_state_dict(src.bn2.state_dict())
            src_ds = getattr(src, "downsample", None)
            if dst.down is not None and src_ds is not None:
                dst.down[0].weight.copy_(src_ds[0].weight)
                dst.down[1].load_state_dict(src_ds[1].state_dict())

        for d_layer, s_layer in [(self.layer1, ref.layer1),
                                 (self.layer2, ref.layer2),
                                 (self.layer3, ref.layer3),
                                 (self.layer4, ref.layer4)]:
            for d_blk, s_blk in zip(d_layer, s_layer):
                copy_block(d_blk, s_blk)

    def forward(self, x):
        x  = self.stem_pad(x)
        x  = self.relu(self.bn1(self.conv1(x)))   # -> S/2
        e1 = self.layer1(x)                        # -> S/2 (skip)
        e2 = self.layer2(e1)                       # -> S/4
        e3 = self.layer3(e2)                       # -> S/8
        e4 = self.layer4(e3)                       # -> S/16 (deep)
        return e1, e4

class CubeResNetSimple(nn.Module):
    """
    ResNet encoder + tiny head (one skip):
    Input : [B*6, 4, S, S]
    Output: [B*6, 1, S, S]
    """
    def __init__(self, in_ch=4, pretrained=True):
        super().__init__()
        self.enc = _ResEnc(in_ch=in_ch, pretrained=pretrained)
        self.fuse = nn.Sequential(
            CPConv3x3(512 + 64, 128),
            CPConv3x3(128, 64),
        )
        self.up_to_S   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine_S  = nn.Sequential(CPConv3x3(64, 32), CPConv3x3(32, 16))
        self.head      = nn.Conv2d(16, 1, 1)
        self.out_act   = nn.Sigmoid()

    def forward(self, x):
        e1, e4 = self.enc(x)
        x = F.interpolate(e4, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.fuse(x)
        x = self.up_to_S(x)
        x = self.refine_S(x)
        return self.out_act(self.head(x))

__all__ = ["CubePad", "CPConv3x3", "CubeResNetSimple"]
