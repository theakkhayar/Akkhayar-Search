from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageStat
import os
import requests
import random
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

SUPABASE_URL = "https://lbyoabqbqpravwjksuiz.supabase.co"
ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxieW9hYnFicXByYXZ3amtzdWl6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY4NTQxMjUsImV4cCI6MjA5MjQzMDEyNX0.gyyopz3sIN83ilqH8bFLHXCKlDHF-iWplf7r38ctJiU"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=ROOT_DIR, static_url_path="")
CORS(app)

MIN_CONFIDENCE_FOR_MATCH = 0.95


def looks_like_text(image: Image.Image) -> bool:
    gray = image.convert("L").resize((128, 128))
    stat = ImageStat.Stat(gray)
    contrast = (stat.stddev[0] or 0) / 255.0
    if contrast < 0.08:
        return False

    mean = stat.mean[0] if stat.mean else 127
    pixels = list(gray.getdata())
    w = 128
    h = 128
    binary = [1 if p < mean else 0 for p in pixels]

    transitions = 0
    for y in range(h):
        row_start = y * w
        prev = binary[row_start]
        for x in range(1, w):
            v = binary[row_start + x]
            if v != prev:
                transitions += 1
                prev = v

    for x in range(w):
        prev = binary[x]
        for y in range(1, h):
            v = binary[y * w + x]
            if v != prev:
                transitions += 1
                prev = v

    denom = (h * (w - 1)) + (w * (h - 1))
    transition_rate = transitions / denom if denom else 0
    if transition_rate < 0.04:
        return False

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(edges.getdata())
    edge_strength = sum(edge_pixels) / (255.0 * len(edge_pixels)) if edge_pixels else 0
    if edge_strength < 0.02:
        return False

    return True


def supabase_query(table, select_cols, filter_col, filter_val):
    try:
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {ANON_KEY}",
            "Content-Type": "application/json"
        }
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        params = {
            "select": select_cols,
            filter_col: f"eq.{filter_val}",
            "limit": 1
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
        return None
    except Exception:
        return None


def init_supabase():
    if not SUPABASE_URL or not ANON_KEY:
        return False
    try:
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {ANON_KEY}"
        }
        response = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers, timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False


supabase_available = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = os.path.join(ROOT_DIR, 'dataset')
class_names = sorted(entry.name for entry in os.scandir(dataset_path) if entry.is_dir())

num_classes = len(class_names)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'my_font_model.pth'), map_location=device))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

MANUAL_FONT_FALLBACKS = {
    "A01_Pixel": {
        "creator_name": "The Akkhayar",
        "social_link": "https://t.me/theakkhayar",
        "status": "Premium",
        "purchase_link": ""
    },
    "A02_Cartoon": {
        "creator_name": "The Akkhayar",
        "social_link": "https://t.me/theakkhayar",
        "status": "Premium",
        "purchase_link": ""
    },
    "A03_Octagon": {
        "creator_name": "The Akkhayar",
        "social_link": "https://t.me/theakkhayar",
        "status": "Premium",
        "purchase_link": ""
    },
}


def get_font_metadata(font_name):
    row = supabase_query("fonts", "creator_name, social_link, status, purchase_link", "font_name", font_name)
    if not row:
        row = supabase_query("fonts", "creator_name, social_link, status, purchase_link", "name", font_name)
    return row if row else {}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': "Missing 'file' field in form data"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        original_img = Image.open(file.stream).convert("RGB")
        if not looks_like_text(original_img):
            return jsonify({
                'font': 'Unknown',
                'confidence': 0.0,
                'creator_name': 'Unknown Creator',
                'creator': 'Unknown Creator',
                'social_link': '',
                'purchase_link': '',
                'status': 'unknown',
                'type': 'free'
            })

        img = preprocess(original_img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_class = class_names[pred.item()]
            confidence = conf.item()

        if confidence < MIN_CONFIDENCE_FOR_MATCH:
            return jsonify({
                'font': 'Unknown',
                'confidence': confidence,
                'creator_name': 'Unknown Creator',
                'creator': 'Unknown Creator',
                'social_link': '',
                'purchase_link': '',
                'status': 'unknown',
                'type': 'free'
            })

        creator_name = "Unknown Creator"
        social_link = ""
        status = "unknown"
        purchase_link = ""
        db_font_name = predicted_class

        if supabase_available:
            row = supabase_query("fonts", "font_name, creator_name, social_link, status, purchase_link", "font_name", predicted_class)
            if not row:
                row = supabase_query("fonts", "font_name, creator_name, social_link, status, purchase_link", "name", predicted_class)

            if row and isinstance(row, dict):
                db_font_name = row.get("font_name") or predicted_class
                creator_name = row.get("creator_name") or "Unknown Creator"
                social_link = row.get("social_link") or ""
                status = str(row.get("status") or "unknown")
                purchase_link = row.get("purchase_link") or ""

        if creator_name == "Unknown Creator":
            fallback_data = MANUAL_FONT_FALLBACKS.get(predicted_class)
            if fallback_data:
                creator_name = fallback_data["creator_name"]
                social_link = fallback_data["social_link"]
                status = fallback_data["status"]
                purchase_link = fallback_data.get("purchase_link", "")

        status_normalized = status.lower()

        return jsonify({
            'font': db_font_name,
            'confidence': confidence,
            'creator_name': creator_name,
            'creator': creator_name,
            'social_link': social_link,
            'purchase_link': purchase_link,
            'status': status,
            'type': 'premium' if status_normalized in ('premium', 'paid', 'pro') else 'free'
        })

    except Exception:
        return jsonify({'error': 'Invalid image or prediction failed'}), 500


@app.route('/get_all_fonts', methods=['GET'])
def get_all_fonts():
    # Fetch 20 fonts and return 6 random ones
    if not supabase_available:
        return jsonify([])
    try:
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {ANON_KEY}",
            "Content-Type": "application/json"
        }
        url = f"{SUPABASE_URL}/rest/v1/fonts"
        params = {
            "select": "font_name, creator_name, social_link, status, purchase_link",
            "limit": 20
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            all_fonts = response.json()
            if not all_fonts:
                return jsonify([])
            
            # Pick 6 random fonts (or fewer if less than 6 available)
            num_to_pick = min(len(all_fonts), 6)
            random_fonts = random.sample(all_fonts, num_to_pick)
            return jsonify(random_fonts)
        return jsonify([])
    except Exception:
        return jsonify([])


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(ROOT_DIR, 'ndex.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
