# ===== Install mmdet3d ops =====
# NOTE: we modified the '1.0.0rc4'
# core/bbox/structures and core/points, 
# consistent with mmdet3d '0.17.0'
pip install -v -e . 

# ===== Compile DFA3D =====
cd packages/DFA3D && bash setup.sh 0 && cd ../..

# ===== Build Voxelization modules =====
python packages/Voxelization/setup.py develop
python packages/Voxelization/setup_v2.py develop

# ===== Compile gsformer GaussianEncoder ops =====
cd projects/RadarGS/mmdet3d_plugin/models/voxel_encoder/gsformer/gaussian_encoder/ops && pip install -e . && cd ../../../../../../../..

# ===== Compile LocalAgg head =====
cd projects/RadarGS/mmdet3d_plugin/models/head/localagg && pip install -e . && cd ../../../../../..
cd projects/RadarGS/mmdet3d_plugin/models/head/localagg_prob && pip install -e . && cd ../../../../../..
cd projects/MonoRaGS/mmdet3d_plugin/models/head/localagg_prob_fast && pip install -e . && cd ../../../../../..

# ===== Compile RCBEVDet Fusion ops =====
cd packages/RCBEVFusion && pip install -e . && cd ../../

# Differentiable_voxelization for gaussian pillar in spconv
cd packages/Differentiable_voxelization && pip install -e . && cd ../..