# infer_video.py
import os, cv2, torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.utils import save_image

from models.video_model import CubeRes18UNet   # << changed import
from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi

# Match the sizes you trained with
EH, EW = 256, 512      # equirect output (2S x 4S)
S      = 128           # cube face size

def infer(video_path, ckpt_path, out_dir="./outputs", in_ch=4, base=32):
	"""
	Args:
	  video_path: path to a .mp4 or a directory of .jpg frames
	  ckpt_path : path to model weights (match CubeRes18UNet(in_ch, base))
	  out_dir   : directory to save saliency PNGs
	  in_ch     : 4 for [RGB3 + diff1]
	  base      : base channels in the UNet decoder (match training)
	"""
	os.makedirs(out_dir, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Build model and load weights
	model = CubeRes18UNet(in_ch=in_ch, base=base, pretrained=False).to(device)
	state = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(state, strict=True)
	model.eval()

	# Projections
	e2c = Equi2Cube(output_width=S, input_h=EH, input_w=EW)
	c2e = Cube2Equi(input_w=S)

	# Resize to the EHxEW used in training
	resize = T.Compose([T.ToPILImage(), T.Resize((EH, EW)), T.ToTensor()])

	# Read frames (dir of JPGs OR a video file)
	frames = []
	if os.path.isdir(video_path):
		files = sorted([f for f in os.listdir(video_path)
						if f.lower().endswith(".jpg") or f.lower().endswith(".png")])
		for f in files:
			frames.append(cv2.imread(os.path.join(video_path, f)))
	else:
		cap = cv2.VideoCapture(video_path)
		while True:
			ret, fr = cap.read()
			if not ret: break
			frames.append(fr)
		cap.release()

	if not frames:
		raise RuntimeError(f"No frames found from: {video_path}")

	prev_rgb = None
	for i, fr_bgr in enumerate(tqdm(frames, desc="infer")):
		# Convert to RGB and resize to EH×EW
		rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
		rgb = (resize(rgb) * 255).permute(1, 2, 0).numpy().astype(np.uint8)

		# Motion channel (1-frame absolute diff, grayscale)
		if prev_rgb is None:
			diff_rgb = np.zeros_like(rgb)
		else:
			d = cv2.absdiff(rgb, prev_rgb)
			g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
			diff_rgb = cv2.merge([g, g, g])
		prev_rgb = rgb.copy()

		# Project to 6 cube faces
		faces_rgb = e2c.to_cube(rgb)
		faces_dif = e2c.to_cube(diff_rgb)

		# Build [6, C, S, S]
		face_tensors = []
		for f in range(6):
			r = torch.from_numpy(faces_rgb[f]).permute(2, 0, 1).float() / 255.0  # [3,S,S]
			m = torch.from_numpy(faces_dif[f][:, :, 0]).unsqueeze(0).float() / 255.0  # [1,S,S]
			x = torch.cat([r, m], dim=0)  # [4,S,S]
			face_tensors.append(x)

		x_faces = torch.stack(face_tensors, dim=0).to(device)  # [6,C,S,S]

		# Forward + project back to equirect
		with torch.no_grad():
			pred_faces = model(x_faces)

			# c2e expects [B,6,C,S,S]
			p = pred_faces.unsqueeze(0)
			sal_e = c2e.to_equi_nn(p)    # [1,1,EH,EW]

		# Save grayscale saliency map
		save_image(sal_e[0].clamp(0, 1).cpu(), os.path.join(out_dir, f"sal_{i:04d}.png"))

if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--video_path", required=True)
	ap.add_argument("--ckpt_path",  required=True)
	ap.add_argument("--output_dir", default="./outputs")
	ap.add_argument("--in_ch", type=int, default=4, help="4 = RGB+diff, 3 = RGB")
	ap.add_argument("--base",  type=int, default=32, help="decoder base channels (match training)")
	args = ap.parse_args()
	infer(args.video_path, args.ckpt_path, args.output_dir, in_ch=args.in_ch, base=args.base)