from __future__ import annotations
import os, cv2, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image

from models.video_model import CubeRes18UNet   
from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi
from grad_cam import GradCAM


def set_inplace_false(module: nn.Module):
    """Flip nn.ReLU(inplace=True) -> inplace=False for module-based ReLUs."""
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

def fx_rewrite_functional_relu(model: nn.Module) -> nn.Module:
    """
    Try to rewrite functional F.relu(..., inplace=True) to inplace=False using torch.fx.
    If tracing fails (dynamic control flow, etc.), returns the original model and prints a warning.
    """
    try:
        import torch.fx as fx
        gm = fx.symbolic_trace(model)

        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is F.relu:
                # normalize kwargs
                inplace = node.kwargs.get("inplace", False)
                if inplace is True:
                    node.kwargs["inplace"] = False
                    changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()
            print("[FX] Rewrote functional F.relu(..., inplace=True) → inplace=False")
        else:
            print("[FX] No functional F.relu(..., inplace=True) found")

        return gm
    except Exception as e:
        print(f"[FX] Warning: FX trace failed ({e}). Proceeding without functional-ReLU rewrite.")
        return model

def find_target_conv_layer3(model: nn.Module) -> tuple[nn.Conv2d, str]:
    """
    Heuristically pick a Conv2d in 'layer3'.
    Preference: the second-to-last Conv2d whose qualified name contains 'layer3'.
    Fallback: last Conv2d in the whole model.
    """
    layer3_convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and "layer3" in name:
            layer3_convs.append((name, m))
    if layer3_convs:
        idx = -2 if len(layer3_convs) >= 2 else -1
        name, mod = layer3_convs[idx]
        return mod, name

    last_name, last_mod = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name, last_mod = name, m
    if last_mod is None:
        raise RuntimeError("Could not find any Conv2d for Grad-CAM target.")
    return last_mod, last_name

class FeatureTap(nn.Module):
    """
    Wraps a model; returns the forward activations of `target_module`
    instead of the model's original output.
    """
    def __init__(self, model: nn.Module, target_module: nn.Module):
        super().__init__()
        self.model = model
        self.target = target_module
        self._feats = None
        self._hook = self.target.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self._feats = output

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feats = None
        _ = self.model(x)   # run full forward; we ignore the real head output
        if self._feats is None:
            raise RuntimeError("FeatureTap: target layer was not hit during forward.")
        return self._feats

    def __del__(self):
        self.remove()

def read_frames(video_path: str):
    frames = []
    if os.path.isdir(video_path):
        files = sorted([f for f in os.listdir(video_path)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        for f in files:
            fr = cv2.imread(os.path.join(video_path, f))
            if fr is not None:
                frames.append(fr)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames found from: {video_path}")
    return frames

def minmax01_bchw(t: torch.Tensor) -> torch.Tensor:
    # per-sample 0..1
    b = t.size(0)
    flat = t.view(b, -1)
    tmin = flat.min(1, keepdim=True)[0].view(b, 1, 1, 1)
    tmax = flat.max(1, keepdim=True)[0].view(b, 1, 1, 1)
    return (t - tmin) / (tmax - tmin + 1e-8)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", required=True, help="ERP .mp4 or folder of frames")
    ap.add_argument("--ckpt_path",  required=True, help="Checkpoint for CubeRes18UNet")
    ap.add_argument("--out_dir",    required=True, help="Output directory for PNGs")
    ap.add_argument("--S", type=int, default=120, help="Cube face size (EH=2S, EW=4S)")
    ap.add_argument("--alpha_prior", type=float, default=0.35, help="Blend weight for CAM prior")
    ap.add_argument("--save_png", action="store_true", help="Save sal_*.png, cam_*.png, final_*.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    S  = args.S
    EH = 2 * S
    EW = 4 * S

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full model + weights
    model = CubeRes18UNet(in_ch=4, pretrained=False).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Build a deepcopy for CAM path and sanitize ReLUs
    cam_base = copy.deepcopy(model).to(device).eval()
    set_inplace_false(cam_base)           # fix nn.ReLU modules
    cam_base = fx_rewrite_functional_relu(cam_base)  # fix functional F.relu(..., inplace=True) if present

    # Pick a layer3 Conv2d for CAM target
    target_conv, target_name = find_target_conv_layer3(cam_base)
    print(f"[GradCAM] Using target layer: {target_name}")

    # Wrap: run the full model but return the tapped features
    cam_model = FeatureTap(cam_base, target_conv).to(device).eval()
    cam_engine = GradCAM(cam_model, target_layer=target_conv)

    # Projections: ERP <-> cube
    e2c = Equi2Cube(output_width=S, input_h=EH, input_w=EW)
    c2e = Cube2Equi(input_w=S)

    # Resize ERP frames to EH×EW
    resize = T.Compose([T.ToPILImage(), T.Resize((EH, EW)), T.ToTensor()])

    frames = read_frames(args.video_path)

    prev_rgb = None
    for i, fr_bgr in enumerate(frames):
        # BGR -> RGB, resize to EH×EW, keep as uint8 for E2C
        rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        rgb = (resize(rgb) * 255).permute(1, 2, 0).numpy().astype(np.uint8)

        # Motion channel = absdiff(prev, cur) -> grayscale
        if prev_rgb is None:
            diff_rgb = np.zeros_like(rgb)
        else:
            d = cv2.absdiff(rgb, prev_rgb)
            g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
            diff_rgb = cv2.merge([g, g, g])
        prev_rgb = rgb.copy()

        # ERP -> 6 faces (S×S) for both RGB and Diff
        faces_rgb = e2c.to_cube(rgb)        # dict(int->HWC)
        faces_dif = e2c.to_cube(diff_rgb)

        # Stack into [6,4,S,S]
        faces = []
        for f in range(6):
            r = torch.from_numpy(faces_rgb[f]).permute(2, 0, 1).float() / 255.0     # [3,S,S]
            m = torch.from_numpy(faces_dif[f][:, :, 0]).unsqueeze(0).float() / 255.0# [1,S,S]
            faces.append(torch.cat([r, m], dim=0))
        x_faces = torch.stack(faces, dim=0).to(device)  # [6,4,S,S]
        assert x_faces.size(0) % 6 == 0, "Batch must be divisible by 6."

        # --- Full model saliency (motion-aware) ---
        with torch.no_grad():
            pred_faces = model(x_faces)               # [6,1,S,S]
            sal_e = c2e.to_equi_nn(pred_faces.unsqueeze(0)).clamp(0, 1)     # [1,1,EH,EW]

        # --- Appearance prior via Grad-CAM (Diff channel zeroed) ---
        x_app = x_faces.clone()
        x_app[:, 3, :, :] = 0.0                       # zero motion channel

        try:
            with torch.enable_grad():
                cam_faces_l3 = cam_engine(x_app)      # [6,1,H',W'] at layer3 res
        except RuntimeError as e:
            # No-crash fallback: appearance prior from ReLU'd channel-average of layer3
            print(f"[WARN] Grad-CAM backward failed: {e}")
            print("[WARN] Falling back to appearance prior = relu(layer3_feats).mean(channel)")
            with torch.no_grad():
                feats = cam_model(x_app)              # [6,C,H',W']
                cam_faces_l3 = F.relu(feats).mean(dim=1, keepdim=True)  # [6,1,H',W']

        cam_faces = F.interpolate(cam_faces_l3, size=(S, S), mode="bilinear", align_corners=True)  # [6,1,S,S]
        cam_e = c2e.to_equi_nn(cam_faces.unsqueeze(0)).clamp(0, 1)  # [1,1,EH,EW]

        sal_e = minmax01_bchw(sal_e)
        cam_e = minmax01_bchw(cam_e)
        final_e = (1.0 - args.alpha_prior) * sal_e + args.alpha_prior * cam_e
        final_e = final_e.clamp(0, 1)

        if args.save_png:
            save_image(sal_e[0].cpu(),   os.path.join(args.out_dir, f"sal_{i:04d}.png"))
            save_image(cam_e[0].cpu(),   os.path.join(args.out_dir, f"cam_{i:04d}.png"))
            save_image(final_e[0].cpu(), os.path.join(args.out_dir, f"final_{i:04d}.png"))

    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print("If you still see an in-place error: search your encoder for functional F.relu(..., inplace=True) and set to False.")

if __name__ == "__main__":
    main()
