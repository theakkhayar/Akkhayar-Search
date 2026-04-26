import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# ၁။ လမ်းကြောင်းများ သတ်မှတ်ခြင်း
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'font_model.pth')
LABELS_PATH = os.path.join(ROOT_DIR, 'models', 'labels.json')

# ၂။ Labels ကို Load လုပ်ခြင်း
try:
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    # Handle both array and dictionary formats
    if isinstance(labels, dict):
        class_names = sorted(labels.keys(), key=lambda x: labels[x])
    else:
        class_names = labels
    print(f"Loaded {len(class_names)} font classes: {class_names}")
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

# ၄။ Image Preprocessing (must match training - includes Grayscale)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
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
        
        # Get font metadata from Supabase
        creator_name = "The Akkhayar"
        social_link = "https://t.me/theakkhayar"
        status = "premium"
        purchase_link = ""
        
        if SUPABASE_URL and ANON_KEY:
            try:
                headers = {
                    "apikey": ANON_KEY,
                    "Authorization": f"Bearer {ANON_KEY}",
                    "Content-Type": "application/json"
                }
                url = f"{SUPABASE_URL}/rest/v1/fonts"
                params = {
                    "select": "creator_name, social_link, status, purchase_link, image_url",
                    "font_name": f"eq.{font_name}",
                    "limit": 1
                }
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        row = data[0]
                        creator_name = row.get("creator_name", creator_name)
                        social_link = row.get("social_link", social_link)
                        status = row.get("status", status)
                        purchase_link = row.get("purchase_link", purchase_link)
            except Exception as e:
                print(f"Supabase query error: {e}")
        
        return jsonify({
            'font': font_name,
            'confidence': 100,
            'creator_name': creator_name,
            'social_link': social_link,
            'status': status,
            'purchase_link': purchase_link,
            'type': 'premium' if status.lower() in ('premium', 'paid', 'pro') else 'free'
        })
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/get_all_fonts', methods=['GET'])
def get_all_fonts():
    """Fetch fonts from Supabase with pagination"""
    if not SUPABASE_URL or not ANON_KEY:
        return jsonify([])
    
    try:
        page = int(request.args.get('page', 1))
        fonts_per_page = 6
        
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {ANON_KEY}",
            "Content-Type": "application/json"
        }
        url = f"{SUPABASE_URL}/rest/v1/fonts"
        params = {
            "select": "font_name, creator_name, social_link, status, purchase_link, image_url"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            all_fonts = response.json()
            if not all_fonts:
                return jsonify({'fonts': [], 'currentPage': page, 'totalPages': 0, 'totalFonts': 0})
            
            total_fonts = len(all_fonts)
            total_pages = (total_fonts + fonts_per_page - 1) // fonts_per_page
            start_index = (page - 1) * fonts_per_page
            end_index = start_index + fonts_per_page
            page_fonts = all_fonts[start_index:end_index]
            
            return jsonify({
                'fonts': page_fonts,
                'currentPage': page,
                'totalPages': total_pages,
                'totalFonts': total_fonts
            })
        else:
            return jsonify({'fonts': [], 'currentPage': page, 'totalPages': 0, 'totalFonts': 0})
    except Exception as e:
        print(f"Error fetching fonts: {str(e)}")
        return jsonify({'fonts': [], 'currentPage': 1, 'totalPages': 0, 'totalFonts': 0})

if __name__ == '__main__':
    # Railway အတွက်ရော local အတွက်ရော အဆင်ပြေအောင် port ကို dynamic ထားထားပါတယ်
    port = int(os.environ.get('PORT', 53709))
    app.run(host='0.0.0.0', port=port, debug=True)
