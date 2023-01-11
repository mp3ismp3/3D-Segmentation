
import torch

import os
import logging
from monai.losses import DiceCELoss
from model.unetr import *
from config import *
from monai.networks.nets import UNETR, UNet, SwinUNETR, VNet, DynUNet
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cfg = cfg()

def build_model():
    if cfg.model_name == 'Unet':
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=cfg.out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act="PRELU",
            norm="BATCH",
            dropout=0.2,
            bias=True,
            dimensions=None,
        ).to(cfg.device)

    elif cfg.model_name == 'UNETR':
        model = UNETR(
            in_channels= 1,
            out_channels=cfg.out_channels,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(cfg.device)

    elif cfg.model_name == 'SwinUNETR':
        model = SwinUNETR(
            spatial_dims=3,
            num_heads=(1, 2, 3, 4),
            img_size=(32, 32, 32),
            attn_drop_rate=0.5,
            in_channels=1,
            out_channels=cfg.out_channels,
            feature_size=36,
        ).to(cfg.device)

    elif cfg.model_name == 'nnUnet':
        model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.out_channels,
        kernel_size=[3],
        strides=(2, 2, 2, 2),
        upsample_kernel_size=[3],
        res_block=True,
        ).to(cfg.device)    
    elif cfg.model_name == 'VNet':
        model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=cfg.out_channels,
            dropout_dim = 3,
        ).to(cfg.device)
    else:
        logging.error('Error model name')

    return model
