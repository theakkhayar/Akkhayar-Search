import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os

# Path to the trained model
model_path = 'my_font_model.pth'
# Path to the dataset folder to get class names
dataset_path = 'dataset'
# Path to the image you want to predict
image_path = '11-01.jpg'

# Load the class names from the dataset folder
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

# Define the same transforms as in train.py but ensure grayscale input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale for ResNet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess the image
image = Image.open(image_path).convert('L')  # Open as grayscale
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
num_classes = len(class_names)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    image = image.to(device)
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_font = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    print(f"Predicted Font Name: {predicted_font}")
    print(f"Confidence Percentage: {confidence_percent:.2f}%")