import os
import glob
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')

class FireDataset(Dataset):
    def __init__(self, fire_dir, non_fire_dir, transform=None):
        self.fire_images = glob.glob(os.path.join(fire_dir, "*.jpg"))
        self.non_fire_images = glob.glob(os.path.join(non_fire_dir, "*.jpg"))
        self.all_images = self.fire_images + self.non_fire_images
        self.labels = [0] * len(self.fire_images) + [1] * len(self.non_fire_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


data_transforms = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomApply([
        transforms.ColorJitter(contrast=0.3),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(180, scale=(0.5, 1.0))
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

fire_dir = "/home/jovyan/Desktop/Data/Train_Data/Fire/"
non_fire_dir = "/home/jovyan/Desktop/Data/Train_Data/Non_Fire/"

dataset = FireDataset(fire_dir, non_fire_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import torch.nn as nn
import torch.nn.functional as F

class FireNet(nn.Module):
    def __init__(self):
        super(FireNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16 * 22 * 22, 10)  
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 16 * 22 * 22) 
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = FireNet().to(device)


import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20): 
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.sigmoid(outputs.squeeze()) >= 0.5 
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')



fire_dir = "/home/jovyan/Desktop/Data/Test_Data/Fire/"
non_fire_dir = "/home/jovyan/Desktop/Data/Test_Data/Non_Fire/"

test_transforms = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = FireDataset(fire_dir, non_fire_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        predicted = torch.round(torch.sigmoid(outputs.squeeze()))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct // total}%')

torch.save(model.state_dict(), '/home/jovyan/Desktop/Data/fire_net2.pth')



