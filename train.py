import os
import shutil
import tempfile
import monai
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from model import *
# monai
from monai.losses import DiceCELoss
from monai.visualize import plot_2d_or_3d_image
from monai.visualize import add_animated_gif,make_animated_gif_summary,blend_images, matshow3d
from monai.visualize import CAM
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.config import print_config
from monai.metrics import DiceMetric, MeanIoU, ROCAUCMetric
from monai.networks.nets import UNETR
from monai.losses import DiceLoss
from monai.data import NibabelWriter
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
# tensorboard
from torch.utils.tensorboard import SummaryWriter, FileWriter

# pytorch
import gc
from torchinfo import summary
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
    
from glob import glob

# from preparedata import get_metadata,path2info
import torch
import torch.nn as nn
from  network import *
from transform.transform import train_transforms, val_transforms
# tqmd
from tqdm import tqdm
# config
from config import cfg
# lr
from optimizer import fetch_scheduler
# loss 

#loader
from prepareloader import prepare_loaders
# from loss_function import loss_function, dice_coef, iou_coef
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL
cfg = cfg()


def plot():
    eval_num = 1
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("epoch Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig("loss.png")


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, num_epochs, scaler):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    # 在訓練最開始之前實例化一個 GradScaler
    torch.set_grad_enabled(True)
    pbar = tqdm(range(len(dataloader)))
    val_it = iter(dataloader)
    step = 0
    # pbar = tqdm(enumerate(dataloader), desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    # run_loss = AverageMeter()
    for itr in pbar:
        batch = next(val_it)
        images = batch["image"].to(device, dtype=torch.float)
        masks = batch["label"].to(device, dtype=torch.float)

        with autocast(enabled=True):
            print(images.size())
            y_pred = model(images)
            # print("預測結果", y_pred)
            loss = loss_function(y_pred, masks)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss /= (itr+1)

        scheduler.step()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"] #當下學習率
        pbar.set_postfix(epoch= f"{epoch}/"f"{num_epochs}", step = f"{itr+1}/"f"{len(pbar)}",train_loss=f"{epoch_loss:0.4f}",lr = f"{current_lr:0.7f}", gpu_mem = f"{mem:0.2f} GB", time = f"{time.time() - start_time:0.2f}s")
        start_time = time.time()
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, current_lr


def valid_one_epoch(model, dataloader, device, epoch, num_epochs):
    model.eval()
    # dice_metric.reset()
    # mean_iou.reset()
    start_time = time.time()
    torch.set_grad_enabled(True)
    pbar = tqdm(range(len(dataloader)))
    val_it = iter(dataloader)
    with torch.no_grad():
        for itr in pbar:
            batch = next(val_it)
            images = batch["image"].to("cpu", dtype=torch.float)
            masks = batch["label"].to("cpu", dtype=torch.float)
            roi_size = cfg.patch_size
            sw_batch_size = 4
            val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model, sw_device="cuda", device="cpu")
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_list = decollate_batch(masks)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            mean_iou(y_pred=val_output_convert, y=val_labels_convert)
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        mean_dice_val = dice_metric.aggregate().item()
        mean_iou_val = mean_iou.aggregate().item()
        dice_metric.reset()
        mean_iou.reset()
        pbar.set_postfix(epoch= f"{epoch}/"f"{num_epochs}",step = f"{itr}/"f"{len(pbar)}", mean_dice_val = f"{mean_dice_val:0.4f}", mean_loss = f"{mean_iou_val:0.4f}", gpu_memory = f"{mem:0.2f}GB", time = f"{time.time() - start_time:0.2f}s")
        start_time = time.time()
    torch.cuda.empty_cache()
    gc.collect

    return mean_dice_val, mean_iou_val, val_outputs_list, images, masks




def run_training(model,train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    scaler = GradScaler(enabled=True)
    writer = SummaryWriter(comment = cfg.model_name)

    if torch.cuda.is_available():
        print("cuda:{}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = -np.inf
    best_epoch = -1
    # history = defaultdict(list) #字典，key不在不會報錯

    for epoch in range(1, num_epochs+1):
        gc.collect()

        #train
        epoch_time = time.time()
        train_loss, lr = train_one_epoch(model, optimizer, scheduler, train_loader, device=cfg.device, epoch=epoch, num_epochs = num_epochs, scaler = scaler)
        epoch_loss_values.append(train_loss)
        print("Final training  {}/{}".format(epoch, num_epochs)," | loss: {:.4f}".format(train_loss)," | time {:.2f}s".format(time.time() - epoch_time))
        
        #val
        epoch_time = time.time()
        mean_dice_val, mean_iou_val, val_outputs, val_images, val_labels = valid_one_epoch(model, val_loader, device=cfg.device, epoch =epoch, num_epochs = num_epochs)
        print("Final validation  {}/{}".format(epoch, num_epochs+1),f" | Valid Dice:{mean_dice_val:0.4f} | Valid miou: {mean_iou_val:0.4f}"," | time {:.2f}s".format(time.time() - epoch_time))
        metric_values.append(mean_dice_val)
        
        # history['Train Loss'].append(train_loss)
        # history['Valid dice Loss'].append(mean_dice_val)
        # history['Valid iou Dice'].append(mean_iou_val)
        # history['lr'].append(lr)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('miou_Loss/val', mean_iou_val, epoch)
        writer.add_scalar('dice_Loss/val', mean_dice_val, epoch)
        writer.add_scalar('lr', lr, epoch)

        # plot_2d_or_3d_image(val_images, epoch, writer, index=0,max_channels=3, tag="validation image")
        # plot_2d_or_3d_image(val_labels, epoch, writer, index=0,max_channels=3, tag="validation label")
        # plot_2d_or_3d_image(val_outputs, epoch, writer, index=0,max_channels=3, tag="validation inference")

        print(f"Valid Dice:{mean_dice_val:0.4f} | Valid miou: {mean_iou_val:0.4f}")

        if mean_iou_val >= best_miou:
            print(f"{c_}Valid Score Improved ({best_miou:0.4f} ---> {mean_iou_val:0.4f})")
            best_miou = mean_iou_val
            best_dice = mean_dice_val
            best_epoch = epoch
            PATH = f"./checkpoints/{cfg.model_name}-best_epoch-{epoch:02d}-miou-{best_miou:0.4f}-dice-{best_dice:0.4f}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")

        PATH = f"./checkpoints/last_epoch-{epoch:02d}.bin"
        torch.save(model.state_dict(), PATH)

        print(); print()
    end = time.time()
    time_elapsed = end - start
    writer.close()
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best miou Score: {:.4f}".format(best_miou))
    print("Best dice Score: {:.4f}".format(best_dice))
    print("Best epoch: {:.4f}".format(best_epoch))



if __name__ == '__main__':

    datasets = "./data_json/train/"
    df = pd.read_csv('./data_3d_info.csv')

    # loss

    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=cfg.smooth_nr, smooth_dr=cfg.smooth_dr, lambda_dice=0.6, lambda_ce=0.4)
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)  
    mean_iou = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    # one hot
    post_label = AsDiscrete(to_onehot=4)
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    fold = 2
    # for fold in range(1):v
    train_loader, val_loader, test_loader, val_ds = prepare_loaders(fold, df, debug=cfg.debug)
    model = build_model()
    summary(model, input_size=(4, 1, 96, 96, 96))
    # CAM(nn_module=model, target_layers="layer4", fc_layers="last_linear")
    torch.backends.cudnn.benchmark = True

    # loss
    epoch_loss_values = []
    metric_values = []

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = fetch_scheduler(optimizer)
    if cfg.load:
        model.load_state_dict(
            torch.load(cfg.checkpoints_dir+"/last_epoch-45.bin", map_location=cfg.device)
        )
        logging.info(f'Model loaded from {cfg.load}')
    run_training(model,train_loader, val_loader, optimizer, scheduler, device=cfg.device, num_epochs=cfg.epochs)
    plot()
