import voxel_ops
import torch
import numpy as np

from pcdet.datasets.processor.data_processor import DataProcessor
from functools import partial

def differentiable_voxelize(self, data_dict=None, config=None):
    if data_dict is None:
        self.differentiable_voxel_generator = None
        grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_size = config.VOXEL_SIZE
        return partial(self.differentiable_voxelize, config=config)

    if self.differentiable_voxel_generator is None:
        self.differentiable_voxel_generator = voxel_ops.Voxelization(
            voxel_size=self.voxel_size,
            # [x, y, z]
            point_cloud_range=self.point_cloud_range,
            # [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points=config.MAX_POINTS_PER_VOXEL,
            # int
            max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            # int
            deterministic=config.DETERMINISTIC,
            # boolean
        )
    
    points = data_dict['points']
    if not isinstance(points, torch.Tensor):
        points_torch = torch.from_numpy(points).cuda()
        if config.RETAIN_GRAPH:
            points_torch.requires_grad_(True)
            
        points = points_torch
        
        
    voxel_output = self.differentiable_voxel_generator(points)
    voxels, coordinates, num_points = voxel_output
    if not data_dict['use_lead_xyz']:
        voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
    if config.RETAIN_GRAPH:
        data_dict['points_gen_voxels'] = points_torch
        data_dict['voxels'] = voxels
    elif config.END_TO_END_MODE:
        data_dict['voxels'] = voxels
    else:
        data_dict['voxels'] = voxels.detach().cpu().numpy()
        
    data_dict['voxel_coords'] = coordinates[:, [2, 1, 0]].detach().cpu().numpy()
    data_dict['voxel_num_points'] = num_points.detach().cpu().numpy()
    return data_dict

setattr(DataProcessor, "differentiable_voxelize", differentiable_voxelize)
