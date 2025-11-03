import torch
import torch.nn as nn

def get_pad_size(lrtd_pad):
    if isinstance(lrtd_pad, int):
        p_l = p_r = p_t = p_d = lrtd_pad
    else:
        p_l, p_r, p_t, p_d = lrtd_pad
    return p_l, p_r, p_t, p_d

class CubePadding(nn.Module):
    def __init__(self, lrtd_pad, use_gpu=True):
        super().__init__()
        if isinstance(lrtd_pad, int):
            self.p_l = self.p_r = self.p_t = self.p_d = lrtd_pad
        else:
            self.p_l, self.p_r, self.p_t, self.p_d = lrtd_pad

    def flip(self, tensor, dim):
        idx = torch.arange(tensor.size(dim)-1, -1, -1, device=tensor.device, dtype=torch.long)
        return tensor.index_select(dim, idx)

    def make_cubepad_edge(self, feat_td, feat_lr):
        td_pad = feat_td.size(3)
        lr_pad = feat_lr.size(4)
        if td_pad > lr_pad:
            return feat_lr.repeat(1, 1, 1, td_pad, 1)
        else:
            return feat_td.repeat(1, 1, 1, 1, lr_pad)

    def forward(self, x):
        assert x.dim() == 5 and x.size(1) == 6, f"CubePadding expects [B,6,C,H,W], got {tuple(x.shape)}"
        p_l, p_r, p_t, p_d = self.p_l, self.p_r, self.p_t, self.p_d
        if p_l == p_r == p_t == p_d == 0:
            return x
        
        # Split faces
        f_back  = x[:, 0]
        f_down  = x[:, 1]
        f_front = x[:, 2]
        f_left  = x[:, 3]
        f_right = x[:, 4]
        f_top   = x[:, 5]

        if p_t != 0:
            _t12    = torch.cat([self.flip(f_top[:, :, :p_t, :], 2).unsqueeze(1),
                                 f_front[:, :, -p_t:, :].unsqueeze(1)], dim=1)
            _t123   = torch.cat([_t12, f_top[:, :, -p_t:, :].unsqueeze(1)], dim=1)
            _t1234  = torch.cat([_t123, f_top[:, :, :, :p_t].permute(0,1,3,2).unsqueeze(1)], dim=1)
            _t12345 = torch.cat([_t1234, self.flip(f_top[:, :, :, -p_t:].permute(0,1,3,2), 2).unsqueeze(1)], dim=1)
            _t123456= torch.cat([_t12345, self.flip(f_back[:, :, :p_t, :], 2).unsqueeze(1)], dim=1)
        if p_d != 0:
            _d12    = torch.cat([self.flip(f_down[:, :, -p_d:, :], 2).unsqueeze(1),
                                 self.flip(f_back[:, :, -p_d:, :], 2).unsqueeze(1)], dim=1)
            _d123   = torch.cat([_d12, f_down[:, :, :p_d, :].unsqueeze(1)], dim=1)
            _d1234  = torch.cat([_d123, self.flip(f_down[:, :, :, :p_d].permute(0,1,3,2), 2).unsqueeze(1)], dim=1)
            _d12345 = torch.cat([_d1234, f_down[:, :, :, -p_d:].permute(0,1,3,2).unsqueeze(1)], dim=1)
            _d123456= torch.cat([_d12345, f_front[:, :, :p_d, :].unsqueeze(1)], dim=1)
        if p_l != 0:
            _l12    = torch.cat([f_right[:, :, :, -p_l:].unsqueeze(1),
                                 self.flip(f_left[:, :, -p_l:, :].permute(0,1,3,2), 3).unsqueeze(1)], dim=1)
            _l123   = torch.cat([_l12, f_left[:, :, :, -p_l:].unsqueeze(1)], dim=1)
            _l1234  = torch.cat([_l123, f_back[:, :, :, -p_l:].unsqueeze(1)], dim=1)
            _l12345 = torch.cat([_l1234, f_front[:, :, :, -p_l:].unsqueeze(1)], dim=1)
            _l123456= torch.cat([_l12345, f_left[:, :, :p_l, :].permute(0,1,3,2).unsqueeze(1)], dim=1)
        if p_r != 0:
            _r12    = torch.cat([f_left[:, :, :, :p_r].unsqueeze(1),
                                 f_right[:, :, -p_r:, :].permute(0,1,3,2).unsqueeze(1)], dim=1)
            _r123   = torch.cat([_r12, f_right[:, :, :, :p_r].unsqueeze(1)], dim=1)
            _r1234  = torch.cat([_r123, f_front[:, :, :, :p_r].unsqueeze(1)], dim=1)
            _r12345 = torch.cat([_r1234, f_back[:, :, :, :p_r].unsqueeze(1)], dim=1)
            _r123456= torch.cat([_r12345, self.flip(f_right[:, :, :p_r, :].permute(0,1,3,2), 3).unsqueeze(1)], dim=1)

        if p_r != 0 and p_t != 0:
            p_tr = self.make_cubepad_edge(_t123456[:, :, :, -p_t:, -1:], _r123456[:, :, :, :1, :p_r])
        if p_t != 0 and p_l != 0:
            p_tl = self.make_cubepad_edge(_t123456[:, :, :, :p_t, :1], _l123456[:, :, :, :1, :p_l])
        if p_d != 0 and p_r != 0:
            p_dr = self.make_cubepad_edge(_d123456[:, :, :, -p_d:, -1:], _r123456[:, :, :, -1:, -p_r:])
        if p_d != 0 and p_l != 0:
            p_dl = self.make_cubepad_edge(_d123456[:, :, :, :p_d, :1], _l123456[:, :, :, -1:, -p_l:])

        out = x

        if p_t != 0:
            t_out = torch.cat([_t123456, out], dim=3)
        else:
            t_out = out
        if p_d != 0:
            td_out = torch.cat([t_out, _d123456], dim=3)
        else:
            td_out = t_out

        if p_l != 0:
            _lp = _l123456
            if 'p_tl' in locals(): _lp = torch.cat([p_tl, _l123456], dim=3)
            if 'p_dl' in locals(): _lp = torch.cat([_lp, p_dl], dim=3)
            tdl_out = torch.cat([_lp, td_out], dim=4)
        else:
            tdl_out = td_out

        if p_r != 0:
            _rp = _r123456
            if 'p_tr' in locals(): _rp = torch.cat([p_tr, _r123456], dim=3)
            if 'p_dr' in locals(): _rp = torch.cat([_rp, p_dr], dim=3)
            tdlr_out = torch.cat([tdl_out, _rp], dim=4)
        else:
            tdlr_out = tdl_out

        return tdlr_out
