import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

class FontDataset(Dataset):
    """Custom dataset for font classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def load_labels(labels_path):
    """Load label mapping from JSON file"""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels

def get_image_paths_and_labels(dataset_path, labels):
    """Get all image paths and their corresponding labels"""
    image_paths = []
    image_labels = []
    
    for folder_name, label_id in labels.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.exists(folder_path):
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(folder_path, image_file)
                    image_paths.append(image_path)
                    image_labels.append(label_id)
    
    return image_paths, image_labels

def create_data_loaders(dataset_path, labels_path, batch_size=32, test_size=0.2, random_state=42):
    """Create training and validation data loaders"""
    
    # Load labels
    labels = load_labels(labels_path)
    print(f"Loaded {len(labels)} font classes")
    
    # Get image paths and labels
    image_paths, image_labels = get_image_paths_and_labels(dataset_path, labels)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the dataset. Please check the dataset path and structure.")
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, image_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=image_labels  # Stratified sampling to maintain class distribution
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels for pretrained models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create datasets
    train_dataset = FontDataset(train_paths, train_labels, transform=transform)
    val_dataset = FontDataset(val_paths, val_labels, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, len(labels)

def get_class_names(labels_path):
    """Get class names from labels file"""
    labels = load_labels(labels_path)
    # Sort by label ID to maintain order
    sorted_labels = sorted(labels.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_labels]

if __name__ == "__main__":
    # Test the data loader
    dataset_path = "dataset"
    labels_path = "labels.json"
    
    try:
        train_loader, val_loader, num_classes = create_data_loaders(dataset_path, labels_path)
        class_names = get_class_names(labels_path)
        
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        
        # Test a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the dataset folder exists with A01-A20 subfolders containing images.")
