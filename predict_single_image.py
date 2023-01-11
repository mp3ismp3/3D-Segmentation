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
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset

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


# in_dir = 'D:/Youtube/Organ and Tumor Segmentation/datasets/Task03_Liver/Data_Train_Test'
# model_dir = 'D:/Youtube/Organ and Tumor Segmentation/results/results'


df = pd.read_csv('./data_3d_info.csv')
fold=1
train_loader, val_loader, val_ds = prepare_loaders(fold, df, debug=cfg.debug)
model = build_model()
model.load_state_dict(torch.load(
    os.path.join(cfg.checkpoints_dir, "Unet-best_epoch-91-miou-0.4308-dice-0.5187.bin")))
model.eval()


sw_batch_size = 4
roi_size = (96, 96, 96)




if __name__ == "__main__" :
    with torch.no_grad():
        test_patient = first(val_loader)
        t_volume = test_patient['image']
        #t_segmentation = test_patient['seg']
        
        test_outputs = sliding_window_inference(t_volume.to(cfg.device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53


        for i in range(144):
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_patient["image"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_patient["label"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.savefig(f"./predict/predict_{i}.png")
