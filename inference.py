import timm
import torch
import cv2
import os
import csv
import argparse
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
###main###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='dataset/test_images/1.jpeg', type=str, help='image path')
    parser.add_argument('--weight', default='exp1/_best1.pt', type=str, help='model trained weight path')
    arg = parser.parse_args()

    model_name = 'efficientnet_b5'
    img_size = 256

    #device
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    #resize test images
    test_transform = A.Compose(
    [
        A.Resize(img_size,img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    #load model from checkpoint
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    checkpoint = torch.load(arg.weight, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    #read image
    if not os.path.isfile(arg.img):
        raise ValueError('image not found')
    image = cv2.imread(arg.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = test_transform(image=image)["image"]
    #test model
    model.eval()
    model.to(device)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    output = model(image)
    output = output.view(output.shape[0], output.shape[1:].numel())
    predictions = (torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy()
    predicted_label = ["Field" if is_field else "Road" for is_field in predictions]
    print('result is', predicted_label)

