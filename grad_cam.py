from __future__ import annotations
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[Union[str, nn.Module]] = None):
        """
        Args:
            model: nn.Module to analyze (e.g., encoder-to-layer3 wrapper).
            target_layer: a module or a dotted name (string) to a Conv2d inside `model`.
                          If None, the last Conv2d found in `model` is used.
        """
        self.model = model
        self.model.eval()

        if isinstance(target_layer, str):
            self.target = self._get_submodule(self.model, target_layer)
        elif isinstance(target_layer, nn.Module):
            self.target = target_layer
        else:
            self.target = self._find_last_conv2d(self.model)
            if self.target is None:
                raise RuntimeError("GradCAM: could not find a Conv2d layer in the given model.")

        self._feats = None
        self._grads = None
        self._fh = self.target.register_forward_hook(self._fwd_hook)
        # Use register_full_backward_hook for broader support
        self._bh = self.target.register_full_backward_hook(self._bwd_hook)

    def remove(self):
        """Remove hooks (call when you are done)."""
        if self._fh is not None:
            self._fh.remove()
            self._fh = None
        if self._bh is not None:
            self._bh.remove()
            self._bh = None

    # --- internal utilities ---

    def _fwd_hook(self, module, inputs, output):
        # DO NOT detach; keep computation graph intact for Grad-CAM
        self._feats = output

    def _bwd_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; take grad wrt forward output (index 0)
        self._grads = grad_output[0]

    def _get_submodule(self, module: nn.Module, path: str) -> nn.Module:
        cur = module
        for p in path.split("."):
            cur = getattr(cur, p)
        return cur

    def _find_last_conv2d(self, module: nn.Module) -> Optional[nn.Conv2d]:
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Grad-CAM for a batch.

        Args:
            x: input tensor for the `model` (e.g., [6, C, S, S] for 6 faces)

        Returns:
            cam: [B, 1, H', W'] normalized to [0,1], where H'×W' is the target layer's spatial size.
        """
        # Clear previous state
        self._feats, self._grads = None, None

        # Enable grad for CAM; we don't need parameter grads to persist
        with torch.enable_grad():
            out = self.model(x)                 # arbitrary tensor
            scalar = out.mean()                 # regression-friendly scalar
            self.model.zero_grad(set_to_none=True)
            scalar.backward(retain_graph=True)  # get grad wrt target feature map

        if self._feats is None or self._grads is None:
            raise RuntimeError("GradCAM: hooks did not capture features/gradients. "
                               "Check that your target layer is in the forward path.")

        feats = self._feats                    # [B, C, H', W']
        grads = self._grads                    # [B, C, H', W']

        # Global average pooling the gradients -> channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        cam = (weights * feats).sum(dim=1, keepdim=True) # [B, 1, H', W']
        cam = F.relu(cam)

        # Per-sample min-max normalize to [0,1]
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cmin = cam_.min(dim=1, keepdim=True)[0]
        cmax = cam_.max(dim=1, keepdim=True)[0]
        cam_norm = (cam_ - cmin) / (cmax - cmin + 1e-8)
        cam_norm = cam_norm.view_as(cam)
        return cam_norm

    def __del__(self):
        self.remove()
