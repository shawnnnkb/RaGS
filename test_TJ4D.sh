CONFIG_PATH=./projects/RadarGS/configs/TJ4D-RadarGS_4x4_24e_detect3d.py
CHECKPOINT_PATH=./projects/RadarGS/checkpoints/detections_on_TJ4D.pth
OUTPUT_NAME=TJ4D-RadarGS
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

# python tools_det3d/test.py \
# --config  $CONFIG_PATH \
# --checkpoint $CHECKPOINT_PATH \
# --eval mAP

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
    # --eval mAP \
    --launcher pytorch ${@:4}
