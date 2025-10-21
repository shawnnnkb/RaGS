# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d_0170.datasets import build_dataloader, build_dataset
from mmdet3d_0170.models import build_detector
from tools_det3d.misc.fuse_conv_bn import fuse_module
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('--config', default='projects/RadarGS/configs/student_configs/vod-RadarGS_votpoint_4x4_20e.py', help='test config file path')
    parser.add_argument('--checkpoint', default='work_dirs/vod-RadarGS_votpoint_4x4_20e/epoch_20.pth', help='checkpoint file')
    parser.add_argument('--samples', default=1296, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = False

    # wandb init after runner set up main process  
    cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    project = args.config.split('/')[-1].split('.')[0]
    if not ('pillar' in project.lower() or 'second' in project.lower()):
        figures_path = os.path.join(cfg.work_dir, 'figures_path')
        os.makedirs(figures_path, exist_ok=True)
        cfg.model.update(meta_info = {'figures_path':figures_path, 'project_name': project})
    
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
