import timm
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import argparse
import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import FieldsVsRoadsDataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='efficientnet_b5', type=str, help='model architecture, in timm.list_models()')
    parser.add_argument('--path', default='dataset', type=str, help='dataset path')
    parser.add_argument('--k', default=5, type=int, help='number of folds')
    arg = parser.parse_args()

    # Training settings
    model_name = arg.arch
    print(model_name)
    img_size = 256
    num_folds = arg.k
    batch_size = 16
    lr = 0.0001
    num_epochs = 5
    #device
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    #datset files
    root_directory = arg.path
    fields_directory = os.path.join(root_directory, "fields")
    roads_directory = os.path.join(root_directory, "roads")
    fields_images_filepaths = sorted([os.path.join(fields_directory, f) for f in os.listdir(fields_directory)])
    roads_images_filepaths = sorted([os.path.join(roads_directory, f) for f in os.listdir(roads_directory)])
    images_filepaths = [*fields_images_filepaths, *roads_images_filepaths]
    correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
    
    #resize
    train_transform = A.Compose(
    [
        A.Resize(img_size,img_size),
        ToTensorV2(),
    ])

    #create dataset
    train_dataset = FieldsVsRoadsDataset(images_filepaths=correct_images_filepaths, transform=train_transform)
    
    #K-fold
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

    #loss function
    criterion = nn.BCEWithLogitsLoss()

    # accuracy and loss values
    folds_acc = []
    folds_f1 = []
    
    #k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(kfold.split(train_dataset)):
        print('fold', fold)

        # Initialize/Reset model
        model = timm.create_model(model_name, pretrained=True, num_classes=1)
        model.to(device)

        # Split dataset into K folds: 1 fold for test, K-1 folds for training
        
        train_subsampler = SubsetRandomSampler(train_index)
        test_subsampler = SubsetRandomSampler(test_index)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler)

        #optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training
        print(f'cross validation with {num_folds} folds')
        for epoch in range(0, num_epochs):
            print('Epoch: ', epoch)
            model.train()
            total_loss = []
            total_correct = []
            l = 0
            for i, (img, target) in enumerate(tqdm(trainloader), start=1):
                optimizer.zero_grad()
                l += len(target)
                img = img.to(device).float()
                target = target.to(device)
                target = target.view(-1, 1).float()
                outputs = model(img)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                # Compute accuracy
                pred = torch.sigmoid(outputs)
                num_correct = ((pred > 0.5) == target).sum().item()
                total_correct.append(num_correct)
                total_loss.append(loss.item())

            print('train_loss: ', sum(total_loss) / len(total_loss))
            print('train_acc: ', sum(total_correct) / l)

        #validation: accuracy and F1-score
        model.eval()
        val_loss = []
        val_correct = []
        tp = 0
        fp = 0
        fn = 0
        l = 0
        with torch.no_grad():
            for i, (img, target) in enumerate(tqdm(testloader), start=1):
                l += len(target)
                img = img.to(device).float()
                target = target.to(device)

                target = target.view(-1, 1)
                outputs = model(img)
                loss = criterion(outputs, target.float())

                pred = torch.sigmoid(outputs)
                num_correct = ((pred > 0.5) == target).sum().item()
                val_correct.append(num_correct)
                val_loss.append(loss.item())
                tp += (torch.logical_and((pred >= 0.5) == 1, target == 1)).sum().item()
                fp += (torch.logical_and((pred >= 0.5) == 1, target == 0)).sum().item()
                fn += (torch.logical_and((pred >= 0.5) == 0, target == 1)).sum().item()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn )
        f1_val = 2 * (precision * recall) / (precision + recall )

        val_total_loss = sum(val_loss) / len(val_loss)
        val_acc = sum(val_correct) / l
        folds_acc.append(val_acc)
        folds_f1.append(f1_val)
        print('val_loss: ', val_loss)
        print('val_acc: ', val_acc)
        print('F1 val: ', f1_val)

    for i in range(num_folds):
        print(f'Fold {i}: acc: {folds_acc[i]},f1: {folds_f1[i]}')
    print('Average acc: ', sum(folds_acc) / len(folds_acc))
    print('Average F1: ', sum(folds_f1) / len(folds_f1))
