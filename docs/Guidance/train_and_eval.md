## Training Details

For all datasets, the final voxel is set to a cube of size 0.32 m, and the image sizes are 800×1280 for VoD, 640×800 for TJ4DRadSet, and 544×960 for OmniHD-Scenes. Anchor size and point cloud range are kept as in \cite{OmniHD}. The models are trained on 4 NVIDIA GeForce RTX 4090 GPUs with a batch size of 4 per GPU. AdamW is used as the optimizer, with 12 epochs for pretraining and 24 epochs for joint training. The weights are available for download. Put all checkpoints under the projects/RadarGS/checkpoints.

## Train

```
tmux new -s your_tmux_name
conda activate RadarGS
bash ./tools_det3d/dist_train.sh config_path 4
# modified detailed settings in dist_train.sh
```

The training logs and checkpoints will be saved under the log_folder、

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the projects/RadarGS/checkpoints.
```
bash test_TJ4D.sh # for evaluating the FINAL_TJ4D.pth on TJ4DRadSet dataset.
bash test_VoD.sh # for evaluating the FINAL_VoD.pth on View-of-delft (VoD) dataset.
```