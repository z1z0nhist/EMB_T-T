from torch.utils.data import Dataset
import cv2
import torch

import torch.nn as nn
import timm

class EMB_Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_dir = df['PATH'].values
        self.labels = df['new_labels'].values
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        if self.transform:
            img = self.transform(image=img)["image"]

        return {'image': img,
                'new_labels': torch.tensor(label, dtype=torch.long),
                'path': img_path}

class EMB_model(nn.Module):
  def __init__(self, model_name,target_size,pretrained=True):
    super(EMB_model, self).__init__()
    self.model = timm.create_model(model_name =model_name, pretrained= pretrained)
    if model_name.find('efficient') != -1:
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, target_size)
    else:
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)
  def forward(self, x):
    x = self.model(x)
    return x