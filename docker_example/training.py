import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import time
def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


############# train one epoch ###################################
def training_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    s_t = time.time()
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        img = data['image'].to(device, dtype=torch.float)
        labels = data['new_labels'].to(device, dtype=torch.long)

        batch_size = img.size(0)

        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()

        if (step + 1) % 1 == 0:
            optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    time_ = time.time() - s_t
    return epoch_loss, time_

def val_epoch(model,dataloader, device, epoch):
    with torch.no_grad():
        model.eval()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:

            img = data['image'].to(device, dtype=torch.float)
            labels = data['new_labels'].to(device, dtype=torch.long)

            batch_size = img.size(0)

            outputs = model(img)
            loss = criterion(outputs, labels)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch, Val_Loss=epoch_loss)
        gc.collect()
    return epoch_loss
