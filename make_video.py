# overlays saliency PNGs (sal_0000.png, ...) on top of the original video

import os, cv2, argparse
from tqdm import tqdm

def make_video(original_video, saliency_dir, output_path, alpha=0.6):
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("[err] cannot open", original_video); return

    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    sal_files = sorted([f for f in os.listdir(saliency_dir) if f.lower().endswith(".png")])
    if not sal_files:
        print("[err] no saliency pngs found in", saliency_dir); return

    for f in tqdm(sal_files, desc="compositing"):
        ok, frame = cap.read()
        if not ok: break
        sal = cv2.imread(os.path.join(saliency_dir, f), cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, (w, h))
        heat = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(frame, 1 - alpha, heat, alpha, 0)
        out.write(blend)

    cap.release(); out.release()
    print("saved:", output_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_video", required=True)
    ap.add_argument("--saliency_dir",   required=True)
    ap.add_argument("--output_path",    default=os.path.join(".", "outputs", "overlay.mp4"))
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()
    make_video(args.original_video, args.saliency_dir, args.output_path, alpha=args.alpha)
