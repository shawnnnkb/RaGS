# CONFIG_PATH=./projects/RadarPillarNet/configs/vod-radarpillarnet_modified_4x1_80e.py
# CHECKPOINT_PATH=./projects/RadarPillarNet/checkpoints/epoch_80.pth
# OUTPUT_NAME=vod-radarpillarnet
# PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

CONFIG_PATH=./projects/RadarGS/configs/vod-RadarGS_4x4_24e_detect3d.py
CHECKPOINT_PATH=./projects/RadarGS/checkpoints/detections_on_VoD.pth
OUTPUT_NAME=vod-RadarGS
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

GPUS="4"
PORT=${PORT:-49500}
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --format-only \
    --eval-options submission_prefix=$PRED_RESULTS \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --launcher pytorch ${@:4}

# python tools_det3d/test.py \
# --format-only \
# --eval-options submission_prefix=$PRED_RESULTS \
# --config $CONFIG_PATH \
# --checkpoint $CHECKPOINT_PATH

python tools_det3d/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
