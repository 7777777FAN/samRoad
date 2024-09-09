from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoad, DataRangeCallback

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_road_GTE",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if dev_run else None
    )


    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True   # 告诉pytorch针对固定大小的输入使用同样的高效卷积算法
    torch.backends.cudnn.enabled = True     # 启用cudnn，这是默认的
    

    net = SAMRoad(config)
        

    train_ds, val_ds = SatMapDataset(config, is_train=True, dev_run=dev_run), SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')


    wandb_logger = WandbLogger()

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler,
        devices=[1]
    )
    
     # Pass the checkpoint path to the fit method if resume argument is provided
    ckpt_path = args.resume if args.resume else None

    if ckpt_path:
        # Load checkpoint manually to modify scheduler
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        param_dicts = []
        
        encoder_params = {
                'params': [p for k, p in net.image_encoder.named_parameters() if 'image_encoder.'+k in net.matched_param_names],
                'lr': config.BASE_LR * config.ENCODER_LR_FACTOR*0.1,
            }
        param_dicts.append(encoder_params)
        
        decoder_params = [{
            'params': [p for p in net.GTE_decoder.parameters()],
            'lr': config.BASE_LR*0.1
            }]
        param_dicts += decoder_params
        
        optimizer = torch.optim.Adam(param_dicts, lr=config.BASE_LR)
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        
        # Assuming the scheduler is a MultiStepLR
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 40])
        scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
        
        # Modify the milestones
        scheduler.milestones = {9, 39, 40}

        # Repackage the states in the checkpoint
        checkpoint['optimizer_states'][0] = optimizer.state_dict()
        checkpoint['lr_schedulers'][0] = scheduler.state_dict()
        

        # Create a new checkpoint file with updated scheduler milestones
        torch.save(checkpoint, 'updated_checkpoint.ckpt')
        ckpt_path = 'updated_checkpoint.ckpt'

    trainer.fit(
        net, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path
    )