# Conda RadarGS Installation

**a. Create a conda virtual environment and activate it.**
```shell
# conda remove --name RadarGS --all
conda create -n RadarGS python=3.10 -y
conda activate RadarGS
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)" # for check

```

**c. Install [gsplat](https://github.com/nerfstudio-project/gsplat).**
```shell
# download from https://docs.gsplat.studio/whl/pt20cu118/gsplat/
pip install packaging numpy==1.24.4 setuptools==69.5.1
pip install packages/gsplat/gsplat-1.5.0+pt20cu118-cp310-cp310-linux_x86_64.whl
python packages/gsplat/rasterization.py # for check
```

**d. Install mmengine mmcv mmdet mmseg [mmdetection3D_zh](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html).**

```shell
pip install -U openmim
pip install mmengine
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html # 1.7.2
pip install mmdet==2.27.0
pip install mmsegmentation==0.29.1
```

**e. Install Neighborhood Attention Transformers following [natten](https://www.shi-labs.com/natten/).**

```shell
pip3 install natten==0.14.6+torch200cu118 -f https://shi-labs.com/natten/wheels
```

**f. Install mmdet3d，[DFA3D](https://github.com/IDEA-Research/3D-deformable-attention) and  [bevpool](https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/setup.py).**

```shell
bash setup.sh
```

**g. Install other packages.**

```shell 
pip install fpsample jaxtyping kornia k3d parrots wandb safetensors==0.3.1 yapf==0.40.1 setuptools==59.5.0 numba==0.61.2 numpy==1.24.4 lyft_dataset_sdk trimesh nuscenes-devkit shapely
# -i https://pypi.tuna.tsinghua.edu.cn/simple # some-package
# pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch_geometric==2.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install torch_cluster==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install spconv-cu116 cumm-cu116===0.4.5

# visualization
conda create -n vis python==3.10
pip install vtk==9.2.6
cd docs/mayavi && python setup.py install && cd ../..
pip install pyqt5==5.15.9 seaborn apptools pyface traitsui configobj pygments
# ImportError: cannot import name 'PythonShell' from 'pyface.api' (/home/bxk/.conda/envs/mayavi_vis/lib/python3.10/site-packages/pyface/api.py), then from .python_shell import PythonShell

```

