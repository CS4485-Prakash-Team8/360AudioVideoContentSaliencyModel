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
		self.p = lrtd_pad
		self.cp = CubePadding(lrtd_pad, use_gpu)

	def forward(self, x):
		b6, c, h, w = x.shape
		assert b6 % 6 == 0, f"CubePad: batch {b6} not divisible by 6"
		N = b6 // 6
		x = x.view(N, 6, c, h, w)
		x = self.cp(x)  # -> [N,6,c,h+2p,w+2p]
		return x.view(N * 6, c, h + 2*self.p, w + 2*self.p)

class CPConv3x3(nn.Module):
	""" CubePad -> 3x3 conv -> BN -> ReLU """
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

# cube-safe ResNet18 encoder blocks
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

class _Res18EncCube(nn.Module):
	"""
	Cube-safe ResNet18 encoder:
	- Stem: 7x7/2 (CubePad(3)), NO maxpool
	- Stages: [2,2,2,2] with strides [1,2,2,2] -> outputs at S/2, S/4, S/8, S/16
	- in_ch: 4 (RGB3 + motion1)
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
			self._load_from_resnet18_imagenet(in_ch)

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
	def _load_from_resnet18_imagenet(self, in_ch):
		# load torchvision resnet18 and transplant weights
		from torchvision.models import resnet18, ResNet18_Weights
		ref = resnet18(weights=ResNet18_Weights.DEFAULT)

		# conv1: widen 3->4 (mean of RGB for the 4th / motion channel)
		w = ref.conv1.weight  # [64,3,7,7]
		new_w = torch.zeros(64, 4, 7, 7, dtype=w.dtype, device=w.device)
		new_w[:, :3] = w
		new_w[:,  3] = w.mean(dim=1)  # motion init = mean RGB
		self.conv1.weight.copy_(new_w)

		self.bn1.load_state_dict(ref.bn1.state_dict())

		# copy residual layers
		def copy_block(dst_blk, src_blk):
			dst_blk.conv1.conv.weight.copy_(src_blk.conv1.weight)
			dst_blk.conv1.bn.load_state_dict(src_blk.bn1.state_dict())
			dst_blk.conv2.conv.weight.copy_(src_blk.conv2.weight)
			dst_blk.conv2.bn.load_state_dict(src_blk.bn2.state_dict())
			if dst_blk.down is not None and getattr(src_blk, "downsample", None) is not None:
				dst_blk.down[0].weight.copy_(src_blk.downsample[0].weight)
				dst_blk.down[1].load_state_dict(src_blk.downsample[1].state_dict())

		for d_layer, s_layer in [(self.layer1, ref.layer1),
								 (self.layer2, ref.layer2),
								 (self.layer3, ref.layer3),
								 (self.layer4, ref.layer4)]:
			for d_blk, s_blk in zip(d_layer, s_layer):
				copy_block(d_blk, s_blk)

	def forward(self, x):
		# stem
		x  = self.stem_pad(x)
		x  = self.relu(self.bn1(self.conv1(x)))    # -> S/2
		e1 = self.layer1(x)                        # -> S/2
		e2 = self.layer2(e1)                       # -> S/4
		e3 = self.layer3(e2)                       # -> S/8
		e4 = self.layer4(e3)                       # -> S/16
		return e1, e2, e3, e4

# U-Net style decoder
class CPBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.c1 = CPConv3x3(in_ch,  out_ch)
		self.c2 = CPConv3x3(out_ch, out_ch)
	def forward(self, x): return self.c2(self.c1(x))

class CubeRes18UNet(nn.Module):
	"""
	Encoder: cube-safe ResNet-18 (ImageNet init), input channels= 4
	Decoder: U-Net with cube-safe convs
	Input : [B*6, C, S, S]
	Output: [B*6, 1, S, S]
	"""
	def __init__(self, in_ch=4, base=32, pretrained=True):
		super().__init__()
		self.enc = _Res18EncCube(in_ch=in_ch, pretrained=pretrained)

		# e1:64 S/2, e2:128 S/4, e3:256 S/8, e4:512 S/16
		self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)   # S/8
		self.dec1 = CPBlock(512 + 256, 256)

		self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)   # S/4
		self.dec2 = CPBlock(256 + 128, 128)

		self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)   # S/2
		self.dec3 = CPBlock(128 +  64,  64)

		self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)   # S
		self.dec4 = CPBlock(64, 32)

		self.head = nn.Conv2d(32, 1, 1)
		self.out  = nn.Sigmoid()

	def forward(self, x):
		e1, e2, e3, e4 = self.enc(x)

		d1 = self.up1(e4)               # to S/8
		d1 = self.dec1(torch.cat([d1, e3], dim=1))

		d2 = self.up2(d1)               # to S/4
		d2 = self.dec2(torch.cat([d2, e2], dim=1))

		d3 = self.up3(d2)               # to S/2
		d3 = self.dec3(torch.cat([d3, e1], dim=1))

		d4 = self.up4(d3)               # to S
		d4 = self.dec4(d4)

		return self.out(self.head(d4))

__all__ = ["CubePad", "CPConv3x3", "CubeRes18UNet"]
