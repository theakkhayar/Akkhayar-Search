# Font Recognition Training Infrastructure

This folder contains all the necessary components to train a font classification model using PyTorch.

## Project Structure

```
training/
├── dataset/              # Font images organized by class (A01-A20)
│   ├── A01/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── A02/
│   └── ...
├── models/               # Trained models will be saved here
│   └── font_model.pth
├── labels.json           # Mapping of class names to numerical IDs
├── data_loader.py        # Dataset loading and preprocessing
├── font_cnn.py           # CNN architectures for font classification
├── train_model.py        # Main training script
├── image_utils.py        # Image processing utilities
└── README.md            # This file
```

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   - Place your font images in the `dataset/` folder
   - Organize images by class in subfolders A01, A02, ..., A20
   - Each subfolder should contain images of that specific font

3. **Dataset Structure Example:**
   ```
   dataset/
   ├── A01/
   │   ├── font_sample_1.jpg
   │   ├── font_sample_2.png
   │   └── ...
   ├── A02/
   │   ├── font_sample_1.jpg
   │   └── ...
   └── ...
   ```

## Training

### Basic Training

Run the training script with default settings:

```bash
cd training
python train_model.py
```

### Custom Training Parameters

You can modify the training parameters in the `main()` function of `train_model.py`:

```python
config = {
    'dataset_path': 'dataset',
    'labels_path': 'labels.json',
    'model_save_path': 'models/font_model.pth',
    'model_type': 'resnet',  # 'cnn' or 'resnet'
    'batch_size': 32,
    'num_epochs': 50,
    'test_size': 0.2,
    'random_state': 42
}
```

### Model Options

1. **Custom CNN (`model_type='cnn'`)**
   - Lightweight custom architecture
   - Faster training
   - Good for smaller datasets

2. **ResNet18 (`model_type='resnet'`)**
   - Pretrained on ImageNet
   - Better performance
   - Recommended for most use cases

## Training Features

- **Automatic Train/Validation Split**: 80/20 split with stratified sampling
- **Data Augmentation**: Random flips, rotations, color jitter for training
- **Learning Rate Scheduling**: Automatic reduction when validation loss plateaus
- **Early Stopping**: Saves best model based on validation accuracy
- **Comprehensive Logging**: Training history plots and classification reports

## Output Files

After training, the following files will be generated:

- `models/font_model.pth`: Best trained model
- `models/classification_report.json`: Detailed classification metrics
- `models/training_history.png`: Training/validation loss and accuracy plots

## Image Processing

The `image_utils.py` module provides:

- **Preprocessing**: Resize, grayscale, normalization
- **Augmentation**: Random transformations for training
- **Quality Enhancement**: Contrast and sharpness improvements
- **Batch Processing**: Handle multiple images efficiently

## Using the Trained Model

Once training is complete, you can use the trained model in your main application:

```python
import torch
from font_cnn import create_model

# Load the trained model
checkpoint = torch.load('models/font_model.pth')
model = create_model('resnet', num_classes=20)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for prediction
with torch.no_grad():
    output = model(preprocessed_image)
    predicted_class = torch.argmax(output, dim=1).item()
```

## Performance Tips

1. **GPU Training**: The script automatically uses GPU if available (CUDA)
2. **Batch Size**: Increase batch size if you have enough GPU memory
3. **Data Augmentation**: Adjust augmentation parameters in `image_utils.py`
4. **Learning Rate**: Modify the learning rate in `train_model.py` if needed

## Troubleshooting

### Common Issues

1. **No Images Found**
   - Check that dataset folder exists with A01-A20 subfolders
   - Ensure images are in supported formats (jpg, png, jpeg, bmp, tiff)

2. **Memory Issues**
   - Reduce batch size in config
   - Use smaller image size in transforms

3. **Poor Performance**
   - Increase number of epochs
   - Try ResNet model instead of custom CNN
   - Add more data augmentation

4. **CUDA Errors**
   - Ensure PyTorch is installed with CUDA support
   - Fall back to CPU training automatically if CUDA not available

## Monitoring Training

During training, you'll see:

- Batch-wise loss updates
- Epoch-wise training and validation metrics
- Learning rate adjustments
- Best model save notifications

The training history is automatically plotted and saved for analysis.

## Next Steps

1. Experiment with different model architectures
2. Fine-tune hyperparameters
3. Add more data augmentation techniques
4. Implement cross-validation for better evaluation
5. Consider ensemble methods for improved accuracy
