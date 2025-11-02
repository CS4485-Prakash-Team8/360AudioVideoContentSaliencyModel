"""
Inference on equirectangular video:
- Read frames, resize to EH×EW
- Build 4ch input (RGB3 + motion1)
- Model outputs [1,EH,EW] saliency
- Save grayscale PNGs to --output_dir
"""

import os, cv2, torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.utils import save_image

from models.video_model import CubeResNetSimple
from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi

EH, EW = 240, 480
S = 120

def infer(video_path, ckpt_path, out_dir="./outputs"):
	os.makedirs(out_dir, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = CubeResNetSimple(in_ch=4, pretrained=False).to(device)
	model.load_state_dict(torch.load(ckpt_path, map_location=device))
	model.eval()

	resize = T.Compose([T.ToPILImage(), T.Resize((EH, EW)), T.ToTensor()])
	e2c = Equi2Cube(output_width=S, input_h=EH, input_w=EW)
	c2e = Cube2Equi(input_w=S)

	# read frames
	frames = []
	if os.path.isdir(video_path):
		files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(".jpg")])
		for f in files:
			frames.append(cv2.imread(os.path.join(video_path, f)))
	else:
		cap = cv2.VideoCapture(video_path)
		while True:
			ret, fr = cap.read()
			if not ret: break
			frames.append(fr)
		cap.release()

	prev = None
	for i, fr in enumerate(tqdm(frames, desc="infer")):
		# equirect RGB (H,W,3) uint8 -> (EH,EW)
		rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
		rgb = (resize(rgb) * 255).permute(1, 2, 0).numpy().astype(np.uint8)

		# motion channel
		if prev is None:
			diff_rgb = np.zeros_like(rgb)
		else:
			d = cv2.absdiff(rgb, prev)
			g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
			diff_rgb = cv2.merge([g,g,g])
		prev = rgb

		# project to 6 cube faces
		faces_rgb = e2c.to_cube(rgb)
		faces_dif = e2c.to_cube(diff_rgb)

		# build [6,4,S,S]
		face_tensors = []
		for f in range(6):
			r = torch.from_numpy(faces_rgb[f]).permute(2,0,1).float()/255.0
			m = torch.from_numpy(faces_dif[f][:,:,0]).unsqueeze(0).float()/255.0
			face_tensors.append(torch.cat([r, m], dim=0))
		x = torch.stack(face_tensors, dim=0).to(device)  # [6,4,S,S]

		with torch.no_grad():
			pred_faces = model(x)
			# project faces back to equirect
			pf = pred_faces.permute(0,2,3,1).cpu().numpy()
			pf_t = torch.from_numpy(pf).permute(0,3,1,2).unsqueeze(0).float().to(device)
			sal_e = c2e.to_equi_nn(pf_t).squeeze(0)       # [1,EH,EW]

		save_image(sal_e.clamp(0,1), os.path.join(out_dir, f"sal_{i:04d}.png"))

if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--video_path", required=True)
	ap.add_argument("--ckpt_path",  required=True)
	ap.add_argument("--output_dir", default="./outputs")
	args = ap.parse_args()
	infer(args.video_path, args.ckpt_path, args.output_dir)
