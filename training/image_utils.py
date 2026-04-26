import os
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class ImageProcessor:
    """Utility class for image processing and augmentation"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        
        # Basic transforms for inference
        self.basic_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Training transforms with augmentation
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size for random cropping
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_path, is_training=False):
        """Process a single image"""
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Apply transforms
            if is_training:
                return self.train_transforms(image)
            else:
                return self.basic_transforms(image)
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def preprocess_for_prediction(self, image):
        """Preprocess image for model prediction"""
        if isinstance(image, str):
            # If image is a path, load it
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            # If image is numpy array, convert to PIL
            image = Image.fromarray(image).convert('L')
        
        # Apply basic transforms
        return self.basic_transforms(image)
    
    def enhance_image(self, image):
        """Enhance image quality for better OCR"""
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        
        # Apply enhancements
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return image
    
    def resize_image(self, image, target_size):
        """Resize image to target size while maintaining aspect ratio"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Calculate new size maintaining aspect ratio
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_image = Image.new('L', target_size, 255)  # White background
        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
        new_image.paste(image, offset)
        
        return new_image
    
    def batch_process_images(self, image_paths, is_training=False):
        """Process multiple images"""
        processed_images = []
        valid_paths = []
        
        for image_path in image_paths:
            processed = self.process_image(image_path, is_training)
            if processed is not None:
                processed_images.append(processed)
                valid_paths.append(image_path)
        
        return processed_images, valid_paths
    
    def save_processed_image(self, image, save_path):
        """Save processed image"""
        if isinstance(image, torch.Tensor):
            # Convert tensor back to PIL image
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Convert to PIL
            image = transforms.ToPILImage()(image)
        
        image.save(save_path)
    
    def analyze_image_quality(self, image_path):
        """Analyze image quality metrics"""
        try:
            image = Image.open(image_path).convert('L')
            img_array = np.array(image)
            
            # Calculate metrics
            metrics = {
                'size': image.size,
                'mean_intensity': np.mean(img_array),
                'std_intensity': np.std(img_array),
                'min_intensity': np.min(img_array),
                'max_intensity': np.max(img_array),
                'contrast_ratio': (np.max(img_array) - np.min(img_array)) / (np.max(img_array) + np.min(img_array) + 1e-6)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return None

def create_sample_images(output_dir="sample_images", num_samples=5):
    """Create sample processed images for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    processor = ImageProcessor()
    
    # This would be used with actual dataset images
    print("Sample image processor created. Use with actual dataset images.")

if __name__ == "__main__":
    # Test image processor
    processor = ImageProcessor()
    
    print("Image processor created successfully!")
    print(f"Target image size: {processor.image_size}")
    print(f"Basic transforms: {processor.basic_transforms}")
    print(f"Training transforms: {processor.train_transforms}")
    
    # Test with sample image if available
    sample_image_path = "dataset/A01/sample.jpg"  # Update with actual path
    if os.path.exists(sample_image_path):
        processed = processor.process_image(sample_image)
        print(f"Sample image processed successfully: {processed.shape}")
    else:
        print(f"Sample image not found at {sample_image_path}")
