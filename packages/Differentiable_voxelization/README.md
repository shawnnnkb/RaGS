# voxelization

Differentiable voxelization operator using Pytorch

Compatibale with OpenPCDet ([https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet))

## Usage

* Voxelization operator
  ```python
  from voxel_ops import Voxelization

  voxel_module = Voxelization(voxel_size=[0.16, 0.16, 4],
                              point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                              max_num_points=32, # number of points per voxel
                              max_voxels=40000,
                              deterministic=True,  # differentiable only if deterministic==True
                              )
  ```
* Work with OpenPCDet
  ```yaml
  - NAME: differentiable_voxelize
  	RETAIN_GRAPH: False
  	END_TO_END_MODE: False
  	DETERMINISTIC: True
  	VOXEL_SIZE: [0.16, 0.16, 4]
  	MAX_POINTS_PER_VOXEL: 32
  	MAX_NUMBER_OF_VOXELS: {
  		'train': 16000,
  		'test': 40000
            }
  ```

## Setup

a. Install  `pytorch` properly.

b. Clone this repository.

```shell
git clone git@github.com:Uzukidd/voxelization.git
```

c. Install `voxel-ops` from source.

```shell
cd ./voxelization
pip install -e .
```

## Usage

See `./test/` for more details
