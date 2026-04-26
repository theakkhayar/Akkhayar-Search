import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import create_data_loaders, get_class_names
from font_cnn import create_model, count_parameters

class FontTrainer:
    """Training class for font classification model"""
    
    def __init__(self, model, train_loader, val_loader, class_names, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs, save_path):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} trainable parameters")
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': self.class_names,
                }, save_path)
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%')
        
        # Generate final classification report
        self.generate_classification_report(val_preds, val_labels, save_path)
        
        # Plot training history
        self.plot_training_history(save_path)
        
        return self.best_val_acc
    
    def generate_classification_report(self, preds, labels, save_path):
        """Generate and save classification report"""
        print('\nClassification Report:')
        print(classification_report(labels, preds, target_names=self.class_names))
        
        # Save classification report
        report = classification_report(labels, preds, target_names=self.class_names, output_dict=True)
        with open(os.path.join(os.path.dirname(save_path), 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_training_history(self, save_path):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(save_path), 'training_history.png'))
        plt.close()

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'dataset_path': r'D:\Font-Data\dataset',
        'labels_path': r'D:\Font-Data\training\labels.json',
        'model_save_path': r'D:\Font-Data\models\font_model.pth',
        'model_type': 'resnet',  # 'cnn' or 'resnet'
        'batch_size': 32,
        'num_epochs': 5,
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create data loaders
        train_loader, val_loader, num_classes = create_data_loaders(
            config['dataset_path'], 
            config['labels_path'], 
            batch_size=config['batch_size'],
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        # Get class names
        class_names = get_class_names(config['labels_path'])
        
        # Create model
        model = create_model(config['model_type'], num_classes=num_classes, pretrained=True)
        model = model.to(device)
        
        print(f"Model created: {config['model_type']}")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        
        # Create trainer
        trainer = FontTrainer(model, train_loader, val_loader, class_names, device)
        
        # Train model
        best_acc = trainer.train(config['num_epochs'], config['model_save_path'])
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Model saved to: {config['model_save_path']}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
