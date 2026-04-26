import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sys
import os

# Add training directory to path to import data_loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))
from data_loader import create_data_loaders

if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data using custom data loader
    dataset_path = 'dataset'
    labels_path = 'training/labels.json'
    
    try:
        train_loader, val_loader, num_classes = create_data_loaders(dataset_path, labels_path)
        print(f"Successfully loaded data with {num_classes} classes")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the dataset folder exists and contains the correct subfolders.")
        exit(1)

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'my_font_model.pth')
    print("Training completed and model saved!")