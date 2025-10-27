# train_cube.py
"""
Train a cube-padded UNet on 360° saliency.
- Projects equirect frames to 6 cube faces (size S=120) with Equi2Cube
- Runs CNN on faces (with Cube Padding)
- Projects prediction back to equirect (240x480) with Cube2Equi
- Computes SphereMSE on equirect (area-weighted)
Run:
  python train_cube.py
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch import optim

from models.video_model import CubeUNet
from smse import SphereMSE
from equi_to_cube import Equi2Cube
from cube_to_equi import Cube2Equi

DATA_DIR   = "./360_Saliency_dataset_2018ECCV"
SAVE_LAST  = "./checkpoints/cube_unet_last.pth"
SAVE_BEST  = "./checkpoints/cube_unet_best.pth"

# equirect size
EH, EW = 240, 480
# cube face size such that to_equi gives EH×EW (2S×4S)
S = 120

# Modify depending on your system
BATCH = 2
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2

# 80/20 train/val split of video IDs from ECCV 2018 dataset
TRAIN_IDS = {
 '217','280','309','266','256','268','295','315','258','245','253','265','242','260','224','296',
 '275','223','278','222','285','286','298','320','238','290','234','294','247','283','232','317',
 '282','251','279','229','292','259','288','311','305','262','257','263','271','240','231','249',
 '230','269','273','293','287','319','254','318','228','270','276','289','252','237','312','225',
 '307','219','241','274','227','221','284','233','236','308','235','297','310','267','302','281'
}
VAL_IDS = {
 '248','299','306','316','239','272','314','246','277','300','220','304','303','226','264','243',
 '250','313','255','291','244','218','261','301'
}

class VRFramesE2C(Dataset):
    def __init__(self, root, ids_set, train=True):
        self.train = train
        self.samples = []
        for vid in sorted(os.listdir(root)):
            if vid not in ids_set: continue
            vpath = os.path.join(root, vid)
            if not os.path.isdir(vpath): continue
            frames = sorted([f for f in os.listdir(vpath) if f.lower().endswith(".jpg")])
            for i, f in enumerate(frames):
                img_path = os.path.join(vpath, f)
                gt_path  = img_path[:-4] + "_gt.npy"
                if os.path.isfile(gt_path):
                    self.samples.append((img_path, gt_path, i))
        self.resize = T.Compose([T.ToPILImage(), T.Resize((EH, EW)), T.ToTensor()])
        self.e2c = Equi2Cube(output_width=S, input_h=EH, input_w=EW)
        self.c2e = Cube2Equi(input_w=S)

    def __len__(self): return len(self.samples)

    def _read_rgb(self, path):
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((EH, EW, 3), np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img_path, gt_path, i = self.samples[idx]
        rgb = self._read_rgb(img_path)
        rgb = (self.resize(rgb)*255).permute(1,2,0).numpy().astype(np.uint8)

        diff_img = None
        if i > 0:
            prev_path = img_path.replace(f"{i+1:04d}.jpg", f"{i:04d}.jpg")
            if os.path.isfile(prev_path):
                prev = self._read_rgb(prev_path)
                prev = (self.resize(prev)*255).permute(1,2,0).numpy().astype(np.uint8)
                d = cv2.absdiff(rgb, prev)
                g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
                diff_img = cv2.merge([g,g,g])
        if diff_img is None:
            diff_img = np.zeros_like(rgb)

        # project to 6 faces
        faces_rgb = self.e2c.to_cube(rgb)
        faces_dif = self.e2c.to_cube(diff_img)

        # stack as [6, 4, S, S]
        x_faces = []
        for f in range(6):
            r = torch.from_numpy(faces_rgb[f]).permute(2,0,1).float()/255.0
            d = torch.from_numpy(faces_dif[f][:,:,0]).unsqueeze(0).float()/255.0
            x_faces.append(torch.cat([r,d], dim=0))
        x_faces = torch.stack(x_faces, dim=0)

        # load GT equirect (1,EH,EW) normalized
        gt = torch.from_numpy(np.load(gt_path)).float()
        if gt.ndim > 2: gt = gt.mean(-1)
        gt = cv2.resize(gt.numpy(), (EW, EH), interpolation=cv2.INTER_AREA)
        gt = torch.from_numpy(gt).float()
        if gt.max() > 1.5: gt = gt/255.0
        m = gt.max()
        if m > 0: gt = gt/m
        gt = gt.unsqueeze(0)

        return x_faces, gt

def train_one_epoch(model, loader, device, loss_fn, optimizer):
    model.train()
    total = 0.0
    for x_faces, gt in tqdm(loader, desc="train"):
        B = x_faces.size(0)
        x = x_faces.view(B*6, 4, S, S).to(device)
        pred_faces = model(x).view(B, 6, 1, S, S)

        # project back to equirect
        pred_e_list = []
        for b in range(B):
            p = pred_faces[b]
            p_np = p.permute(0,2,3,1).detach().cpu().numpy()
            p_t = torch.from_numpy(p_np).permute(0,3,1,2).unsqueeze(0).float().to(device)
            e = loader.dataset.c2e.to_equi_nn(p_t)
            pred_e_list.append(e.squeeze(0))
        pred_e = torch.stack(pred_e_list, dim=0).to(device)

        gt = gt.to(device)
        loss = loss_fn(pred_e, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total/len(loader)

@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    for x_faces, gt in tqdm(loader, desc="val"):
        B = x_faces.size(0)
        x = x_faces.view(B*6, 4, S, S).to(device)
        pred_faces = model(x).view(B,6,1,S,S)

        pred_e_list = []
        for b in range(B):
            p_np = pred_faces[b].permute(0,2,3,1).detach().cpu().numpy()
            p_t = torch.from_numpy(p_np).permute(0,3,1,2).unsqueeze(0).float().to(device)
            e = loader.dataset.c2e.to_equi_nn(p_t)
            pred_e_list.append(e.squeeze(0))
        pred_e = torch.stack(pred_e_list, dim=0).to(device)

        gt = gt.to(device)
        loss = loss_fn(pred_e, gt)
        total += loss.item()
    return total/len(loader)

def main():
    os.makedirs("./checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_ds = VRFramesE2C(DATA_DIR, TRAIN_IDS, train=True)
    val_ds   = VRFramesE2C(DATA_DIR, VAL_IDS,   train=False)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CubeUNet(in_ch=4, base=32).to(device)
    criterion = SphereMSE(EH, EW).to(device)
    optimiz   = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)

    best = float("inf")
    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch(model, train_ld, device, criterion, optimiz)
        va = validate(model, val_ld, device, criterion)
        print(f"[E{epoch}] train {tr:.4f}  val {va:.4f}")
        torch.save(model.state_dict(), SAVE_LAST)
        if va < best:
            best = va
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"[INFO] new best {best:.4f}")

if __name__ == "__main__":
    main()
