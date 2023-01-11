# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from utils.utils import dice, resample_3d

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

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
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoints_dir, "SwinUNETR-best_epoch-01-miou-0.6162-dice-0.7152.bin")))
    model.eval()

    datasets = "./data_json/train/"
    df = pd.read_csv('./data_3d_info.csv')
    # print(df)
    # for fold in range(5):
    fold = 4
    train_loader, val_loader, test_loader, test_ds = prepare_loaders(fold, df, debug=cfg.debug)

    train_df = pd.read_csv('./new_data.csv')

    with torch.no_grad():
        case_num = 3
        # for i in range(200): 
        #     img_name = os.path.split(test_ds[i]['image'].meta["filename_or_obj"])[1]
        #     print("名稱",img_name)
        img = test_ds[case_num]["image"]
        label = test_ds[case_num]["label"]
        img_name = os.path.split(test_ds[case_num]['image'].meta["filename_or_obj"])[1]
        # group_spacing = group[1][["px_spacing_h"]].values[0][0]
        group_affine = np.eye(4) * 1.5
        # img_name = test_ds["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        print(label.shape)
        _, h, w, d = label.shape

        target_shape = (h, w, d)
        test_inputs = torch.unsqueeze(img, 1).cuda()
        test_labels = torch.unsqueeze(label, 1).cuda()  
        test_outputs = sliding_window_inference(
            test_inputs, (96, 96, 96), 4, model, overlap=0.8
        )
        test_outputs = torch.softmax(test_outputs, 1).cpu().numpy()
        test_outputs = np.argmax(test_outputs, axis=1).astype(np.uint8)[0]
        test_labels = test_labels.cpu().numpy()[0, 0, :, :, :]
        test_outputs = resample_3d(test_outputs, target_shape)
        nib.save(
            nib.Nifti1Image(test_outputs.astype(np.uint8), group_affine), os.path.join("./predict/", img_name)
        )