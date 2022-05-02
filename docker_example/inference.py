import argparse
import os

import pandas as pd
import torch
from Module import EMB_model,EMB_Dataset
from make_csv import make_csv_file
from tqdm import tqdm

import cv2
from torch.utils.data import Dataset, DataLoader
parser = argparse.ArgumentParser(description= 'inference code')
parser.add_argument('--file_path', type = str, help = 'inference file path')
args = parser.parse_args()
# model_detail = 'C:/Users/ikh/PycharmProjects/docker_example/best_model/best.csv'
model_detail = './best_model/best.csv'
best_pd = pd.read_csv(model_detail,index_col = 0).transpose()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import albumentations as A
from albumentations.pytorch import transforms
from sklearn.preprocessing import LabelEncoder
def data_transforms_img(img_size):
    data_transforms = {
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
if __name__ == "__main__":

    encoder = LabelEncoder()

    total = 0
    correct = 0
    name = []
    co = []
    predict = []
    model = EMB_model(model_name=best_pd['model_name'][0], target_size=6)
    # infer_path = args.file_path
    infer_path =  'C:/Users/ikh/Downloads/test-20220420T054816Z-001/test/'
    infer_df = make_csv_file(infer_path)
    infer_df['new_labels'] = encoder.fit_transform(infer_df['labels'])

    data_transforms = data_transforms_img(int(best_pd['image_size'][0]))
    infer = EMB_Dataset(infer_df, transforms=data_transforms['valid'])
    valid_loader = DataLoader(infer, batch_size=8, num_workers=1, shuffle=True, pin_memory=True, drop_last=True)
    with torch.no_grad():
        best_model = '{}/{}best{}_{}_{}'.format(best_pd['model_name'][0], best_pd['model_name'][0], best_pd['epoch'][0], best_pd['image_size'][0],best_pd['date'][0])

        model.load_state_dict(torch.load(best_model))
        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()

        bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, data in bar:
            img = data['image'].to(device, dtype=torch.float)
            labels = data['new_labels'].to(device, dtype=torch.long)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # if predicted != labels:
            #     name.append(data['path'][0].split('/')[-1])
            #     co.append(encoder.classes_[labels.cpu()])
            #     predict.append(encoder.classes_[predicted.cpu()])
            correct += (predicted == labels).sum().item()
    #
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # pd.DataFrame({'image_name': name, 'predcited': predict, 'labels': co, 'results': predict == co})