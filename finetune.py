# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import models_vit
import util.misc as misc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from iopath.common.file_io import g_pathmgr as pathmgr
from engine_finetune import evaluate, train_one_epoch
import util.lr_decay as lrd

from util.kinetics import Kinetics
from util.logging import master_print as print
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument("--batch_size_per_gpu", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--save_prefix", default="", type=str, help="prefix for saving checkpoint and log files")

    # Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--img_size", default=224, type=int, help="images input size")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop_path_rate", type=float, default=0.1, metavar="PCT", help="Drop path rate")
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--layer_decay", type=float, default=0.8, help="layer-wise lr decay from ELECTRA/BEiT")
    parser.add_argument("--min_lr", type=float, default=1e-5, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--warmup_epochs", type=int, default=0, metavar="N", help="epochs to warmup LR")

    # Augmentation parameters
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT", help="Color jitter factor (enabled only when not using Auto/RandAug)")
    parser.add_argument("--aa", type=str, default="rand-m7-mstd0.5-inc1", metavar="NAME", help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")

    # * Finetuning params
    parser.add_argument("--resume", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="Use class token instead of global pool for classification")
    parser.add_argument("--num_classes", default=700, type=int, help="number of the classes")
    parser.add_argument("--train_dir", default="", help="path to train data")
    parser.add_argument("--val_dir", default="", help="path to val data")
    parser.add_argument("--datafile_dir", type=str, default="./datafiles", help="Store data files here")
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--jitter_scales_relative", default=[0.2, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+")
    parser.add_argument("--train_jitter_scales", default=[256, 320], type=int, nargs="+")

    return parser

def list_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.path)
    subdirectories.sort()  # Sort the list of subdirectories alphabetically
    return subdirectories

def find_mp4_files(directory):
    """Recursively search for .mp4 or .webm files in a directory"""
    mp4_files = []
    subdir_idx = 0
    subdirectories = list_subdirectories(directory)
    for subdir in subdirectories:
        files = os.listdir(subdir)
        files.sort()
        for file in files:
            if file.endswith((".mp4", ".webm")):
                mp4_files.append((os.path.join(subdir, file), subdir_idx))
        subdir_idx += 1
    return mp4_files

def write_csv(video_files, save_dir, save_name):
    """Write the .csv file with video path and subfolder index"""
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as csvfile:
        for video_file, subdir_idx in video_files:
            csvfile.write(f"{video_file}, {subdir_idx}\n")

def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = Kinetics(
        mode="finetune",
        datafile_dir=args.datafile_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=tuple(args.train_jitter_scales),
        train_crop_size=args.img_size,
        repeat_aug=args.repeat_aug,
        pretrain_rand_erase_prob=args.reprob,
        pretrain_rand_erase_mode=args.remode,
        pretrain_rand_erase_count=args.recount,
        pretrain_rand_erase_split=args.resplit,
        aa_type=args.aa,
        rand_aug=args.rand_aug,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )
    dataset_val = Kinetics(
        mode="val",
        datafile_dir=args.datafile_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=tuple(args.train_jitter_scales),
        train_crop_size=args.img_size,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    print(f"Sampler_train = {sampler_train}")

    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=8*args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # define model
    model = models_vit.__dict__[args.model](**vars(args))
    model.to(device)
    model_without_ddp = model
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    # optionally compile model
    if args.compile:
        model = torch.compile(model)

    # effective batch size
    eff_batch_size = (args.batch_size_per_gpu * args.accum_iter * misc.get_world_size() * args.repeat_aug)
    print(f"Effective batch size: {eff_batch_size} = {args.batch_size_per_gpu} batch_size_per_gpu * {args.accum_iter} accum_iter * {misc.get_world_size()} GPUs * {args.repeat_aug} repeat_augs")

    # effective lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print(f"Effective lr: {args.lr}")

    # wrap model in ddp
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # can use any other scheduler here
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # build criterion
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    print(f"Criterion = {criterion}")

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, args=args)
        test_stats = evaluate(data_loader_val, model_without_ddp, device)
        print(f"Accuracy of the model on the {len(dataset_val)} test images (top-5): {test_stats['acc5']:.1f}%")

        if args.output_dir and test_stats["acc5"] > max_accuracy:
            print("Improvement in max test accuracy. Saving model!")
            checkpoint_path = misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy = max(max_accuracy, test_stats["acc5"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch}

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(f"{args.output_dir}/{args.save_prefix}_log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # increment lr scheduler
        scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return [checkpoint_path]

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # prepare data files
    train_files = find_mp4_files(directory=args.train_dir)
    write_csv(video_files=train_files, save_dir=args.datafile_dir, save_name='train')
    val_files = find_mp4_files(directory=args.val_dir)
    write_csv(video_files=val_files, save_dir=args.datafile_dir, save_name='val')

    # finetune
    main(args)