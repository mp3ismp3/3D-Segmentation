from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
    EnsureChannelFirstd,
    EnsureTyped,

)

from monai import transforms
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
from monai import data
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
import numpy as np
from config import cfg
from monai.inferers import sliding_window_inference
from network import build_model
from prepareloader import prepare_loaders
from functools import partial
import nibabel as nib
import cv2 

import time

        
if __name__ == "__main__" :
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoints_dir, "SwinUNETR-best_epoch-122-miou-0.5171-dice-0.6088.bin")))
    model.eval()

    datasets = "./data_json/train/"
    df = pd.read_csv('./data_3d_info.csv')
    fold = 4
    train_loader, val_loader, test_loader, test_ds = prepare_loaders(fold, df, debug=cfg.debug)

    with torch.no_grad():
        case_num = 0
        # for i in range(200): 
        #     img_name = os.path.split(test_ds[i]['image'].meta["filename_or_obj"])[1]
        #     print("名稱",img_name)
        img = test_ds[case_num]["image"]
        label = test_ds[case_num]["label"]
        test_inputs = torch.unsqueeze(img, 1).cuda()
        test_labels = torch.unsqueeze(label, 1).cuda()
        start = time.time()
        print(test_inputs.size)
        test_outputs = sliding_window_inference(
            test_inputs, (96, 96, 96), 4, model, overlap=0.8
        )
        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

        for i in range(429):
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title("image")
            plt.imshow(test_inputs.cpu().numpy()[0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(test_labels.cpu().numpy()[0, 0, :, :, i], cmap="hot")
            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(
                torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, i], cmap="hot"
            )
            plt.savefig(f"./predict/predict_{i}.png")
