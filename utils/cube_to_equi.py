import numpy as np
import torch
from torch import nn
from utils.sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from utils.sph_utils import face_to_cube_coord, norm_to_cube

class Cube2Equi:
    def __init__(self, input_w):
        in_width = input_w
        out_w = in_width * 4
        out_h = in_width * 2

        XX, YY = np.meshgrid(range(out_w), range(out_h))
        theta, phi = xy2angle(XX, YY, out_w, out_h)
        theta = pruned_inf(theta)
        phi = pruned_inf(phi)

        _x, _y, _z = to_3dsphere(theta, phi, 1)
        face_map = np.zeros((out_h, out_w))
        face_map = get_face(_x, _y, _z, face_map)
        x_o, y_o = face_to_cube_coord(face_map, _x, _y, _z)

        out_coord = np.transpose(np.array([x_o, y_o]), (1, 2, 0))
        out_coord = norm_to_cube(out_coord, in_width)

        self.out_coord = out_coord.astype(np.float32)
        self.face_map  = face_map.astype(np.int64)

    def to_equi_nn(self, input_data):
        assert input_data.dim() == 5 and input_data.size(1) == 6, \
            f"Cube2Equi expects [B,6,C,S,S], got {tuple(input_data.shape)}"
        device = input_data.device
        dtype  = input_data.dtype

        gridf = torch.as_tensor(self.out_coord, device=device)
        face_map = torch.as_tensor(self.face_map, device=device)

        out_h, out_w = gridf.size(0), gridf.size(1)
        B, _, C, S, _ = input_data.size()

        warp_out = torch.zeros((B, C, out_h, out_w), dtype=dtype, device=device, requires_grad=True)

        g = (gridf - gridf.max()/2) / (gridf.max()/2)
        g = g.unsqueeze(0).repeat(B, 1, 1, 1)

        for f in range(6):
            mask = (face_map == f)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, C, out_h, out_w)

            sampled = nn.functional.grid_sample(input_data[:, f], g, align_corners=True)
            warp_out = torch.where(mask, sampled, warp_out)

        return warp_out
