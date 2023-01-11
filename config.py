import torch
import os
from glob import glob

class cfg:
    seed          = 101
    load          = True
    inference     = False
    in_channels   = 1
    out_channels  = 4
    patch_size    = (96, 96, 96)
    debug         = False # set debug=False for Full Training
    checkpoints_dir = './checkpoints'
    model_name    = 'SwinUNETR'
    pretrained_model_name = 'large_unet_fold0_0.9024.pth'
    batch_size    = 1
    train_bs      = 32
    valid_bs      = train_bs*2
    epochs        = 100
    lr            = 1e-4
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = 100
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    num_classes   = 4
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    smooth_dr     = 1e-5
    smooth_nr     = 1e-5