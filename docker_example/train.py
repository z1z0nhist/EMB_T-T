import os
import warnings
import sys

import pandas as pd
import numpy as np

import mlflow
from tqdm import tqdm
import argparse
import torch
import logging
import torch.optim as optim
import time
import copy
from collections import defaultdict
import gc
from torch.optim import lr_scheduler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from training import training_epoch,val_epoch
from itertools import product
from make_csv import make_csv_file
from Module import EMB_model,EMB_Dataset
from shutil import copyfile
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import transforms

parser = argparse.ArgumentParser(description= 'train code')
parser.add_argument('--train_path', type = str, help = 'inference file path')
parser.add_argument('--valid_path', type = str, help = 'inference file path')
args = parser.parse_args()

config = config = {
    "model_name" : ['tf_efficientnet_b5_ns','resnet18','resnet50'],
    "epoch" : [1,2],
    "img_size" : [224,448]
}

now = datetime.now()

def data_transforms_img(img_size):
    data_transforms = {
        "train": A.Compose([
            A.Resize(img_size, img_size),
            A.ShiftScaleRotate(shift_limit=0.1,
                               scale_limit=0.15,
                               rotate_limit=60,
                               p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()]),

        "valid": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()])
    }
    return data_transforms


def fetch_scheduler(optimizer):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,eta_min=1e-6)
    return scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

total = sorted(list(product(*list(config.values()))), key = lambda x:(x[0],x[2]))

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss,train_time = training_epoch(model, optimizer, scheduler,
                                    dataloader=Train_loader,
                                    device=device, epoch=epoch)
        val_epoch_loss = val_epoch(model,dataloader=valid_loader,
                                   device=device, epoch=epoch)

        history['model_name'].append(model_name)
        history['Train Loss'].append(train_epoch_loss)
        history['Val Loss'].append(val_epoch_loss)
        history['epoch'].append(epoch)
        history['image_size'].append(im_szie)
        history['training_time'].append(train_time)
        history['date'].append("{}_{}_{}.bin".format(now.year,now.month,now.day))
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            # PATH = "{}/Loss{:.4f}_epoch{:.0f}.bin".format(model_name, best_epoch_loss, epoch)
            # if not os.path.isdir('{0}/'.format(model_name)):
            #     os.mkdir('{0}/'.format(model_name))
            # torch.save(model.state_dict(), PATH)
            # # Save a model file from the current directory
            # print(f"Model Saved")
        print()
        #
        # end = time.time()
        # time_elapsed = end - start
        # print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        # print("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

    PATH = "{}/{}best{}_{}_{}_{}_{}.bin".format(model_name,model_name,best_epoch,im_szie,now.year,now.month,now.day)
    if not os.path.isdir('{0}/'.format(model_name)):
        os.mkdir('{0}/'.format(model_name))
    torch.save(model.state_dict(), PATH)
    pd.DataFrame(history).to_csv("{}/{}best{}_{}_{}_{}_{}_log.csv".format(model_name,model_name,epoch,im_szie,now.year,now.month,now.day))
    return model, history , pd.DataFrame(history).iloc[best_epoch-1]

if __name__ == "__main__":
    result = []
    # train_path = 'C:/Users/ikh/Downloads/train-20220420T054817Z-001/train'
    # valid_path = 'C:/Users/ikh\Downloads/test-20220420T054816Z-001/test'
    train_path = args.train_path
    valid_path = args.valid_path

    train_df = make_csv_file(train_path)
    valid_df = make_csv_file(valid_path)

    encoder = LabelEncoder()
    train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
    valid_df['new_labels'] = encoder.fit_transform(valid_df['labels'])

    target_size = len(encoder.classes_)
    for i in total:
        model_name = i[0]
        epoch = i[1]
        im_szie = i[2]

        data_transforms = data_transforms_img(im_szie)

        model = EMB_model(model_name=model_name, target_size=target_size)
        if torch.cuda.is_available():
            model = model.to(device)
        Train = EMB_Dataset(train_df, transforms=data_transforms['train'])
        Train_loader = DataLoader(Train, batch_size=8, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)

        valid = EMB_Dataset(valid_df, transforms=data_transforms['valid'])
        valid_loader = DataLoader(valid, batch_size=8, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)

        optimizer = optim.Adam(model.parameters(), lr=0.0001,
                               weight_decay=1e-6)
        scheduler = fetch_scheduler(optimizer)

        torch.cuda.empty_cache()
        model, history, best_value = run_training(model, optimizer, scheduler, device=device, num_epochs=epoch)

        result.append(best_value)
    if not os.path.isdir('{0}/'.format('best_model')):
        os.mkdir('{0}/'.format('best_model'))
    pandas_result = pd.DataFrame(result).sort_values(by=['Train Loss', 'training_time'])
    pandas_result.to_csv("{}/{}_{}_{}_{}_log.csv".format('best_model', 'best_model', now.year, now.month, now.day))
    pandas_result[['model_name','epoch','image_size','date']].loc[0].transpose().to_csv("{}/best.csv".format('best_model'))

