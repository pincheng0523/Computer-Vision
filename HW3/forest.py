import os
import torch
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')

def predict_image(image_path, model, device):
    model.to(device)  
    model.eval()  
    
    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device) 
    output = model(image)
    prediction = torch.round(torch.sigmoid(output.squeeze()))  
    return 'Fire' if prediction.item() == 0 else 'Non Fire'


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

model = FireNet()
model.load_state_dict(torch.load('/home/jovyan/Desktop/Data/fire_net2.pth', map_location=device))
model.to(device).eval()

'''
image_path = '/home/jovyan/Desktop/4.jpeg'  
print(predict_image(image_path, model, device))
'''
'''
def display_prediction(image_path, model, device):
    prediction = predict_image(image_path, model, device)
    image = Image.open(image_path)
    
    plt.imshow(image)
    plt.axis('off')  
    plt.title(f'Prediction: {prediction}')
    plt.show()

display_prediction(image_path, model, device)
'''
def display_folder_predictions(folder_path, model, device):
    images = os.listdir(folder_path)
    
    for image_name in images:
        if image_name.lower().endswith('.jpeg'):
            image_path = os.path.join(folder_path, image_name)
            prediction = predict_image(image_path, model, device)
            
            image = Image.open(image_path)
            plt.imshow(image)
            plt.axis('off')  
            plt.title(f'Prediction: {prediction}')
            plt.show()
            
images_folder = '/home/jovyan/Desktop/foresttest/'  
display_folder_predictions(images_folder, model, device)

