# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.utils import PFNLayer
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
from torch.nn import functional as F

class PFNLayer_Radar(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        self.in_channels1 = 8
        self.in_channels2 = 2
        self.in_channels3 = 2
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units1 = out_channels // 2
        self.units2 = out_channels // 4
        self.units3 = out_channels // 4
        self.norm1 = build_norm_layer(norm_cfg, self.units1)[1]  # 需要归一化的维度为32
        self.norm2 = build_norm_layer(norm_cfg, self.units2)[1]# 需要归一化的维度为15
        self.norm3 = build_norm_layer(norm_cfg, self.units3)[1]# 需要归一化的维度为16
        self.linear1 = nn.Linear(self.in_channels1, self.units1, bias=False)
        self.linear2 = nn.Linear(self.in_channels2, self.units2, bias=False)
        self.linear3 = nn.Linear(self.in_channels3, self.units3, bias=False)
        assert mode in ['max', 'avg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        spatio_input = inputs.index_select(2,torch.tensor([0,1,2,5,6,7,8,9]).to(inputs.device))
        velocity_input = inputs.index_select(2,torch.tensor([3,10]).to(inputs.device))
        snr_input = inputs.index_select(2,torch.tensor([4,11]).to(inputs.device))
        x1 = self.linear1(spatio_input)
        x1 = self.norm1(x1.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()  # N,C 或者 N C L
        x2 = self.linear2(velocity_input)
        x2 = self.norm2(x2.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x3 = self.linear3(snr_input)
        x3 = self.norm3(x3.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                             1).contiguous()
        x = torch.cat((x1,x2,x3),dim=-1)
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # 0是数，1是索引
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer_Radar_logits(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        self.num_classes = num_classes
        self.in_channels1 = 8
        self.in_channels2 = 2
        self.in_channels3 = 2
        self.in_channels4 = num_classes + 1
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units1 = out_channels // 4
        self.units2 = out_channels // 4
        self.units3 = out_channels // 4
        self.units4 = out_channels // 4
        self.norm1 = build_norm_layer(norm_cfg, self.units1)[1]  #
        self.norm2 = build_norm_layer(norm_cfg, self.units2)[1]  # 
        self.norm3 = build_norm_layer(norm_cfg, self.units3)[1]  #
        self.norm4 = build_norm_layer(norm_cfg, self.units4)[1]  # 
        self.linear1 = nn.Linear(self.in_channels1, self.units1, bias=False)
        self.linear2 = nn.Linear(self.in_channels2, self.units2, bias=False)
        self.linear3 = nn.Linear(self.in_channels3, self.units3, bias=False)
        self.linear4 = nn.Linear(self.in_channels4, self.units4, bias=False)
        assert mode in ['max', 'avg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        if self.num_classes == 3: logits_select_idx = [12,13,14,15]
        if self.num_classes == 4: logits_select_idx = [12,13,14,15,16]
        spatio_input = inputs.index_select(2,torch.tensor([0,1,2,5,6,7,8,9]).to(inputs.device))
        velocity_input = inputs.index_select(2,torch.tensor([3,10]).to(inputs.device))
        snr_input = inputs.index_select(2,torch.tensor([4,11]).to(inputs.device))
        logits_input = inputs.index_select(2,torch.tensor(logits_select_idx).to(inputs.device))
        x1 = self.linear1(spatio_input)
        x1 = self.norm1(x1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  # N,C 或者 N C L
        x2 = self.linear2(velocity_input)
        x2 = self.norm2(x2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x3 = self.linear3(snr_input)
        x3 = self.norm3(x3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x4 = self.linear4(logits_input)
        x4 = self.norm4(x4.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = torch.cat((x1,x2,x3,x4),dim=-1)
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # 0是数，1是索引
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

@VOXEL_ENCODERS.register_module()
class Radar7PillarVFE(nn.Module):

    def __init__(self,
                 in_channels=7,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):  ##  false为默认的
        super(Radar7PillarVFE, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center  # 点中心
        self._with_voxel_center = with_voxel_center  # voxel中心
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True  # 只有一层
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)  # 每个非空pillar的均值xyz
            f_cluster = features[:, :, :3] - points_mean  # 所有点和均值的差包括补充的零点
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])  # 这里coor好像是bzyx
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +  # ：，：，3可以相减，对应每个voxel中心，包括补充的0点
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]   # 这里是trick，改变了前两维特征变成了局部xy，相对于voxel中心
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # 最里面一维接起来（xyzsv1v2tXcYcZcXpYp）[X,10,12]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]  # maxpoints
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)  # 接起来的特征有的是0点的，需要把这些再归成0
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  #  这里无效点特征都会是0

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()

@VOXEL_ENCODERS.register_module()
class RadarPillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True, ##  false为默认的
                 with_velocity_snr_center=True):
        super(RadarPillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        if with_velocity_snr_center:
            in_channels += 2
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center  # 点中心
        self._with_voxel_center = with_voxel_center  # voxel中心
        self._with_velocity_snr_center = with_velocity_snr_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True  # 只有一层
            pfn_layers.append(
                PFNLayer_Radar(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)  # 每个非空pillar的均值xyz
            f_cluster = features[:, :, :3] - points_mean  # 所有点和均值的差包括补充的零点
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])  # 这里coor好像是bzyx
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +  # ：，：，3可以相减，对应每个voxel中心，包括补充的0点
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]   # 这里是trick，改变了前两维特征变成了局部xy，相对于voxel中心
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        if self._with_velocity_snr_center:
            velocity_snr_mean = features[:, :, 3:5].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            velocity_snr_center = features[:, :, 3:5] - velocity_snr_mean
            features_ls.append(velocity_snr_center)
        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # 最里面一维接起来（xyzvrXcYcZcXpYpVcRc）[X,10,12]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]  # maxpoints
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)  # 接起来的特征有的是0点的，需要把这些再归成0
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  #  这里无效点特征都会是0

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()

@VOXEL_ENCODERS.register_module()
class RadarPillarFeatureNet_logits(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 num_classes=3,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True, ##  false为默认的
                 with_velocity_snr_center=True):
        super(RadarPillarFeatureNet_logits, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        if with_velocity_snr_center:
            in_channels += 2
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center  # 点中心
        self._with_voxel_center = with_voxel_center  # voxel中心
        self._with_velocity_snr_center = with_velocity_snr_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True  # 只有一层
            pfn_layers.append(
                PFNLayer_Radar_logits(
                    in_filters,
                    out_filters,
                    num_classes,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)  # 每个非空pillar的均值xyz
            f_cluster = features[:, :, :3] - points_mean  # 所有点和均值的差包括补充的零点
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])  # 这里coor好像是bzyx
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +  # ：，：，3可以相减，对应每个voxel中心，包括补充的0点
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]   # 这里是trick，改变了前两维特征变成了局部xy，相对于voxel中心
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        if self._with_velocity_snr_center:
            velocity_snr_mean = features[:, :, 3:5].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            velocity_snr_center = features[:, :, 3:5] - velocity_snr_mean
            features_ls.append(velocity_snr_center)
        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # 最里面一维接起来（xyzvrXcYcZcXpYpVcRc）[X,10,12]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]  # maxpoints
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)  # 接起来的特征有的是0点的，需要把这些再归成0
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  #  这里无效点特征都会是0

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()