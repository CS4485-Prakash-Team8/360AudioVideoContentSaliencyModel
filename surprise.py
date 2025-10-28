import numpy as np

# Positive z-score with moving window (causal-ish, symmetric pad).
def _zscore_pos(x: np.ndarray, win: int = 5) -> np.ndarray:
    x = x.astype(np.float32)
    if len(x) < 3:
        return np.zeros_like(x, dtype=np.float32)
    # simple moving average and variance
    kernel = np.ones(win, dtype=np.float32) / float(win)
    mu  = np.convolve(x, kernel, mode='same')
    var = np.convolve((x - mu) ** 2, kernel, mode='same')
    z = (x - mu) / np.sqrt(np.maximum(var, 1e-8))
    z[z < 0] = 0.0
    return z

# Compute surprise signal from per-frame saliency grids.
def surprise_from_saliency_grid(arr_T_H_W: np.ndarray, win: int = 5) -> np.ndarray:
    # use mean absolute change of grid probs as a simple volatility signal
    T = arr_T_H_W.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float32)
    # frame-to-frame absolute change, averaged over tiles
    dif = np.abs(arr_T_H_W[1:] - arr_T_H_W[:-1]).mean(axis=(1, 2))
    dif = np.concatenate([[0.0], dif], axis=0)  # same length T
    z = _zscore_pos(dif, win=win)
    return z.astype(np.float32)

# Apply surprise gating to per-frame saliency grids.
def apply_gating(arr_T_H_W: np.ndarray, z_T: np.ndarray, beta: float = 0.4) -> np.ndarray:
    assert arr_T_H_W.ndim == 3
    assert len(z_T) == arr_T_H_W.shape[0]
    gated = (1.0 + beta * z_T[:, None, None]) * arr_T_H_W
    denom = gated.sum(axis=(1, 2), keepdims=True) + 1e-8
    return (gated / denom).astype(np.float32)


# toy prob sequence
T,H,W = 6,8,16
arr = np.ones((T,H,W), np.float32) / (H*W)
arr[3,2:5,7:10] += 0.01  # inject a “surprise” bump at t=3
arr /= arr.sum(axis=(1,2), keepdims=True)

z = surprise_from_saliency_grid(arr, win=3)
arr2 = apply_gating(arr, z, beta=0.5)

print("z:", z.round(3))
print("sum(arr2[t]) ->", (arr2.sum(axis=(1,2))).round(6))  # should all be ~1.0
