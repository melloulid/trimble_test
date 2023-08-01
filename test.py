import timm
import torch
import cv2
import os
import csv
import argparse
from torchvision import transforms
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#test dataset
class FieldsVsRoadsInferenceDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

#show images with the lables
def display_image_grid(save_folder, images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 15))
    for i, image_filepath in enumerate(images_filepaths):
        
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted_label = predicted_labels[i]
        color = "green"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.plot()
    plt.savefig(os.path.join(save_folder, 'test_results.png'))

###main###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/test_images', type=str, help='test data path')
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
    test_images_filepaths = sorted([os.path.join(arg.data, f) for f in os.listdir(arg.data)])
    test_dataset = FieldsVsRoadsInferenceDataset(images_filepaths=test_images_filepaths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_images_filepaths), shuffle=False)
    #load model from checkpoint
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    checkpoint = torch.load(arg.weight, map_location=device) 
    model.load_state_dict(checkpoint["state_dict"])

    #test model
    model.eval()
    model.to(device)
    predicted_labels = []
    with torch.no_grad():
      for images in test_loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        output = output.view(output.shape[0], output.shape[1:].numel())
        predictions = (torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy()
        predicted_labels += ["Field" if is_field else "Road" for is_field in predictions]
    #save images with their predictions
    display_image_grid('dataset',test_images_filepaths, predicted_labels)
    print('predictions saved')
    
