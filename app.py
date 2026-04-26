import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ၁။ လမ်းကြောင်းများ သတ်မှတ်ခြင်း
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'font_model.pth')
LABELS_PATH = os.path.join(ROOT_DIR, 'models', 'labels.json')

# ၂။ Labels ကို Load လုပ်ခြင်း
try:
    with open(LABELS_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} font classes.")
except Exception as e:
    print(f"Error loading labels: {e}")
    class_names = []

# ၃။ Model အား အသစ်ပြန်လည် တည်ဆောက်ခြင်း
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

if os.path.exists(MODEL_PATH):
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing without model weights...")
model.to(device)
model.eval()

# ၄။ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return send_from_directory(ROOT_DIR, 'ndex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            font_name = class_names[predicted.item()]
            
        print(f"Predicted Font: {font_name}")
        return jsonify({'font': font_name})
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    # Railway အတွက်ရော local အတွက်ရော အဆင်ပြေအောင် port ကို dynamic ထားထားပါတယ်
    port = int(os.environ.get('PORT', 53709))
    app.run(host='0.0.0.0', port=port, debug=True)