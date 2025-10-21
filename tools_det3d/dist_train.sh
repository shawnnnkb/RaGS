# TORCH_DISTRIBUTED_DEBUG=DETAIL
# tmux new -s train3d
# conda activate RadarGS

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500} # if using multi-exp should change PORT
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config $CONFIG \
    --launcher pytorch ${@:3}
    
# NOTE: remind train epochs in config file
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_12e_pretrain.py 4 > vod-RadarGS_4x4_12e_pretrain.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_12e_detect3d.py 4 > vod-RadarGS_4x4_12e_detect3d.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_6400.py 4 > vod-RadarGS_4x4_06e_ablation_6400.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_10800.py 4 > vod-RadarGS_4x4_06e_ablation_10800.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_3200.py 4 > vod-RadarGS_4x4_06e_ablation_3200.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_3x4_06e_ablation_19200.py 4 > vod-RadarGS_3x4_06e_ablation_19200.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_3x4_06e_ablation_25600.py 4 > vod-RadarGS_3x4_06e_ablation_25600.log 2>&1 &

# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_decoder3.py 4 > vod-RadarGS_4x4_06e_ablation_decoder3.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_decoder2.py 4 > vod-RadarGS_4x4_06e_ablation_decoder2.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_decoder1.py 4 > vod-RadarGS_4x4_06e_ablation_decoder1.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_1600.py 4 > vod-RadarGS_4x4_06e_ablation_1600.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_fusionnone.py 4 > vod-RadarGS_4x4_06e_ablation_fusionnone.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_3x4_06e_ablation_fusion3.py 4 > vod-RadarGS_3x4_06e_ablation_fusion3.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_mvxyz.py 4 > vod-RadarGS_4x4_06e_ablation_mvxyz.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_IMAradarpl.py 4 > vod-RadarGS_4x4_06e_ablation_IMAradarpl.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_3DDCA.py 4 > vod-RadarGS_4x4_06e_ablation_3DDCA.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_wodepth.py 4 > vod-RadarGS_4x4_06e_ablation_wodepth.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_woradar.py 4 > vod-RadarGS_4x4_06e_ablation_woradar.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_wofrustum.py 4 > vod-RadarGS_4x4_06e_ablation_wofrustum.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_gaussianformer.py 4 > vod-RadarGS_4x4_06e_ablation_gaussianformer.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/ablation_anchor_num/vod-RadarGS_4x4_06e_ablation_gaussianformer_resume.py 4 > vod-RadarGS_4x4_06e_ablation_gaussianformer_resume.log 2>&1 &


# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_24e_detect3d.py 4 > vod-RadarGS_4x4_24e_detect3d.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_24e_detect3d_resume.py 4 > vod-RadarGS_4x4_24e_detect3d_resume.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_24e_detect3d_pruning.py 4 > vod-RadarGS_4x4_24e_detect3d_pruning.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarGS/configs/vod-RadarGS_4x4_24e_detect3d_resume24.py 4 > vod-RadarGS_4x4_24e_detect3d_resume24.log 2>&1 &
