import torch
import torch.nn as nn
import torch.nn.functional as F

class FontCNN(nn.Module):
    """CNN architecture for font classification"""
    
    def __init__(self, num_classes=20, input_channels=3):
        super(FontCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling operations, 224x224 becomes 14x14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

class ResNetFontClassifier(nn.Module):
    """ResNet-based classifier for better performance"""
    
    def __init__(self, num_classes=20, pretrained=True):
        super(ResNetFontClassifier, self).__init__()
        
        # Load pretrained ResNet18
        from torchvision import models
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the final fully connected layer for our number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

def create_model(model_type='cnn', num_classes=20, pretrained=True):
    """Create a model of the specified type"""
    
    if model_type == 'cnn':
        return FontCNN(num_classes=num_classes)
    elif model_type == 'resnet':
        return ResNetFontClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """Test the model with dummy input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CNN model
    model = FontCNN(num_classes=20).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"CNN Model output shape: {output.shape}")
    print(f"CNN Model parameters: {count_parameters(model):,}")
    
    # Test ResNet model
    resnet_model = ResNetFontClassifier(num_classes=20).to(device)
    
    with torch.no_grad():
        resnet_output = resnet_model(dummy_input)
    
    print(f"ResNet Model output shape: {resnet_output.shape}")
    print(f"ResNet Model parameters: {count_parameters(resnet_model):,}")

if __name__ == "__main__":
    test_model()
