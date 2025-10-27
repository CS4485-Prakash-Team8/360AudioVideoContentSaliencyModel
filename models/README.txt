360° Saliency with Cube-Padded U-Net + Sphere-Weighted MSE

Trains a 2D U-Net on the six faces of a cubemap derived from 360° (equirectangular) video frames, using Cube Padding to make 3×3 convolutions topologically consistent across face boundaries. Predictions are mapped back to equirectangular space and optimized with Sphere-weighted MSE (SMSE) to respect pixel area on the sphere.

Model: U-Net (RGB + simple motion channel) operating on cube faces

Padding: CubePadding

Loss: SMSE (area-weighted MSE on sphere)

Projection: Equi2Cube (equirect → cube), Cube2Equi (cube → equirect)

Dataset: 360_Saliency_dataset_2018ECCV (Install and unzip to ./360_Saliency_dataset_2018ECCV)

You can use /checkpoints/cube_unet_best.pth as most recent checkpoint or train model from scratch

Portions of this implementation are adapted (with attribution) from prior work listed under Citations.

Citations:

Sphere-weighted MSE (SMSE) & U-Net baseline
C. H. Vo, J.-C. Chiang, D. H. Le, T. T. A. Nguyen, T. V. Pham,
Saliency Prediction for 360-degree Video, GTSD 2020, pp. 442–448.
DOI: 10.1109/GTSD50082.2020.9303135
Code reference (SMSE): smse.py from
https://github.com/vhchuong/Saliency-prediction-for-360-degree-video

@INPROCEEDINGS{9303135,
  author={C. H. {Vo} and J.-C. {Chiang} and D. H. {Le} and T. T. A. {Nguyen} and T. V. {Pham}},
  booktitle={2020 5th International Conference on Green Technology and Sustainable Development (GTSD)},
  title={Saliency Prediction for 360-degree Video},
  year={2020},
  pages={442-448},
  doi={10.1109/GTSD50082.2020.9303135}
}


Cube Padding & Spherical utilities
F.-Y. Chao, C. Ozcinar, L. Zhang, W. Hamidouche, O. Deforges, A. Smolic,
Towards Audio-Visual Saliency Prediction for Omnidirectional Video with Spatial Audio, VCIP 2020, pp. 355–358.
DOI: 10.1109/VCIP49819.2020.9301766
Code reference (CubePadding & sph_utils):
https://github.com/FannyChao/AVS360_audiovisual_saliency_360

@INPROCEEDINGS{9301766,
  author={F.-Y. {Chao} and C. {Ozcinar} and L. {Zhang} and W. {Hamidouche} and O. {Deforges} and A. {Smolic}},
  booktitle={2020 IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  title={Towards Audio-Visual Saliency Prediction for Omnidirectional Video with Spatial Audio},
  year={2020},
  pages={355-358},
  doi={10.1109/VCIP49819.2020.9301766}
}


Dataset (ECCV 2018)
Z. Zhang, Y. Xu, J. Yu, S. Gao,
Saliency Detection in 360° Videos, ECCV 2018.
Dataset and reference code:
https://github.com/xuyanyu-shh/Saliency-detection-in-360-video

@InProceedings{Zhang_2018_ECCV,
  author    = {Zhang, Ziheng and Xu, Yanyu and Yu, Jingyi and Gao, Shenghua},
  title     = {Saliency Detection in 360° Videos},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month     = {September},
  year      = {2018}
}

#1 install deps
pip install torch torchvision numpy opencv-python tqdm pillow

#2 arrange dataset under ./360_Saliency_dataset_2018ECCV (frames + *_gt.npy)

#3 train
python train_video.py

#4 infer
python infer_video.py --video_path path/to/video.mp4 --ckpt_path checkpoints/cube_unet_best.pth --output_dir outputs/infer

#5 make overlay video
python make_video.py --original_video path/to/video.mp4 --saliency_dir outputs/infer --output_path outputs/overlay.mp4

