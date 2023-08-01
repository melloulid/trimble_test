import timm
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import numpy as np
import collections
from collections import defaultdict
from dataset import FieldsVsRoadsDataset

#training function
def train(train_loader, model, criterion, optimizer, epoch, device):
    total_correct=[]
    total_loss = []
    model.to(device)
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
      optimizer.zero_grad()
      images = images.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True).float().view(-1, 1)
      output = model(images)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      pred = torch.sigmoid(output)
      correct = ((pred > 0.5) == target).sum().item()
      total_correct.append(correct)
      total_loss.append(loss.item())

    train_loss = sum(total_loss) / len(total_loss)
    train_acc = sum(total_correct) / len(train_loader.dataset)
    return train_loss, train_acc
#validation function
def validate(val_loader, model, criterion, epoch, device):
    total_correct=[]
    total_loss = []
    tp =0
    fp =0
    fn = 0
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            pred = torch.sigmoid(output)
            correct = ((pred > 0.5) == target).sum().item()
            total_correct.append(correct)
            total_loss.append(loss.item())
            tp += (torch.logical_and((pred >= 0.5) == 1, target == 1)).sum().item()
            fp += (torch.logical_and((pred >= 0.5) == 1, target == 0)).sum().item()
            fn += (torch.logical_and((pred >= 0.5) == 0, target == 1)).sum().item()
    val_loss = sum(total_loss) / len(total_loss)
    val_acc = sum(total_correct) / len(val_loader.dataset)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return val_loss, val_acc, f1

###main###    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='dataset', help='dataset path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-perc', default=0.3, type=float, help='validation set percentage')
    parser.add_argument('--exp', type=str, default='exp1', help='experiment name')
    arg = parser.parse_args()

    #training parameters
    model_name = 'efficientnet_b5'
    img_size = 256
    epochs = arg.epochs
    batch_size = arg.batch_size
    #device
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    #dataset files
    root_directory = arg.path
    fields_directory = os.path.join(root_directory, "fields")
    roads_directory = os.path.join(root_directory, "roads")
    fields_images_filepaths = sorted([os.path.join(fields_directory, f) for f in os.listdir(fields_directory)])
    roads_images_filepaths = sorted([os.path.join(roads_directory, f) for f in os.listdir(roads_directory)])
    images_filepaths = [*fields_images_filepaths, *roads_images_filepaths]
    correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
    random.seed(42)
    random.shuffle(correct_images_filepaths)
    total_number = len(correct_images_filepaths)
    val_index = int(arg.val_perc*total_number)
    val_images_filepaths = correct_images_filepaths[0:val_index]
    train_images_filepaths = correct_images_filepaths[val_index:total_number]
    #data augmentation for training
    train_transform = A.Compose(
    [
        A.Resize(img_size,img_size),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=250, width=250),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    #resize for validation
    val_transform = A.Compose(
    [
        A.Resize(img_size,img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    #dataset construction
    val_dataset = FieldsVsRoadsDataset(images_filepaths=val_images_filepaths, transform=val_transform)
    train_dataset = FieldsVsRoadsDataset(images_filepaths=train_images_filepaths, transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f'{len(train_loader.dataset)} images in training set')
    print(f'{len(val_loader.dataset)} images in validation set')
    #model cration
    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    

    #loss function, optimizer
    #weighted loss for unbalanced dataset
    weight = torch.tensor([2.0]).to(device) 
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight= weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #learning history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    #if the results are not enhanced after 20 epochs 
    #training will be stopped and the model with the lowest val loss is saved
    best_val_loss = math.inf
    early_stopping = collections.deque(maxlen=int(20))
    model_save_dir=arg.exp
    os.makedirs(model_save_dir, exist_ok=True)
    #start training
    for epoch in range(epochs):
      print(f'Epoch {epoch}')
      train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, epoch, device)
      print('train_loss: ', train_loss)
      print('train_accuracy: ', train_accuracy)
      train_losses.append(train_loss)
      train_accuracies.append(train_accuracy)

      val_loss, val_accuracy, val_f1 = validate(val_loader, model, criterion, epoch, device)
      print('val_loss: ', val_loss)
      print('val_accuracy: ', val_accuracy)
      print('f1: ', val_f1)
      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)
      
      # save if better on val_loss
      if val_loss < best_val_loss:
        model_save_path = os.path.join(model_save_dir , "_best{}".format(epoch) + ".pt")
        best_val_loss = val_loss

        save_dict = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict()
            }
        torch.save(save_dict, model_save_path)
        
      early_stopping.append(val_loss)
      if len(early_stopping) >= early_stopping.maxlen:
        # if all elements in list are higher than best_val_loss
        if all(np.array(early_stopping)> best_val_loss):
            model_save_path = os.path.join(model_save_dir,  "_e{}".format(epoch) + ".pt")
            
            save_dict = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
                }
            torch.save(save_dict, model_save_path)
            print('training finished')

            break
        else:
            index_max = np.argmin(np.array(early_stopping))
        

        #loss curve
        fig = plt.figure(figsize=(20, 10))
        plt.title("loss curve")
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.xlabel('num_epochs', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(model_save_dir, 'loss_curve.png'))

        #accuracy curve
        fig = plt.figure(figsize=(20, 10))
        plt.title("accuracy curve")
        plt.plot(train_accuracies, label='train_acc')
        plt.plot(val_accuracies, label='val_acc')
        plt.xlabel('num_epochs', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(model_save_dir, 'accuracy_curve.png'))

