#Importing the Libraries
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms

# Preprocess images by resizing them
def preprocess_images():
    for file_name in os.listdir():
        name, extension = os.path.splitext(file_name)
        if extension == '.png':
            original_image = cv2.imread(file_name)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(original_image, (100, 100))
            cv2.imwrite(file_name, resized_image)

# Function to get all the PNG file names
def get_png_filenames():
    return [file_name for file_name in os.listdir() if file_name.endswith('.png')]

# Save the filenames to CSV
def save_filenames_to_csv(file_array):
    class_dictionary = {
        'file_name': file_array
    }
    df = pd.DataFrame(class_dictionary)
    df.to_csv('dataset.csv')

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = plt.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# Neural Network Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10368, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x

# Inference function
def inference(model, test_loader, file_array):
    for values in test_loader:
        output = model(values)
        pred = output.argmax(dim=1, keepdim=True)

    shape_names = ["Circle", "Square", "Octagon", "Heptagon", "Nonagon", "Star", "Hexagon", "Pentagon", "Triangle"]
    for count, output_value in enumerate(pred):
        print(f"File Name: {file_array[count]}")
        print(shape_names[output_value.item()])

if __name__ == '__main__':
    preprocess_images()
    file_array = get_png_filenames()
    save_filenames_to_csv(file_array)

    root_directory = os.getcwd()
    transform = transforms.ToTensor()
    dataset = CustomDataset(csv_file="dataset.csv", root_dir=root_directory, transform=transform)
    batch_size = 500
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('model.pt'))
    
    inference(model, test_loader, file_array)