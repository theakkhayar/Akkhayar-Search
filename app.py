import os
import json
import logging
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import Lock
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageStat, ImageOps
import requests
import random
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load .env file if it exists, but don't fail if it doesn't
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info("Loaded .env file")
else:
    logger.info("No .env file found, using environment variables")

# Production-ready environment variable handling
SUPABASE_URL = os.environ.get("SUPABASE_URL")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Validate required environment variables
if not SUPABASE_URL:
    logger.warning("SUPABASE_URL not set - database features will be disabled")
if not ANON_KEY:
    logger.warning("SUPABASE_ANON_KEY not set - database features will be disabled")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=ROOT_DIR, static_url_path="")
CORS(app)

MIN_CONFIDENCE_FOR_MATCH = 0.95

torch.set_grad_enabled(False)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


@app.errorhandler(404)
def handle_404(_error):
    if request.path.startswith(("/predict", "/get_all_fonts")):
        return jsonify({"error": "Not found"}), 404
    return "Not found", 404


def looks_like_text(image: Image.Image) -> bool:
    gray = image.convert("L").resize((128, 128))
    stat = ImageStat.Stat(gray)
    contrast = (stat.stddev[0] or 0) / 255.0
    if contrast < 0.08:
        return False

    mean = stat.mean[0] if stat.mean else 127
    w = 128
    h = 128
    pix = gray.tobytes()
    if not pix:
        return False

    transitions = 0
    for y in range(h):
        row_start = y * w
        prev = 1 if pix[row_start] < mean else 0
        for x in range(1, w):
            v = 1 if pix[row_start + x] < mean else 0
            if v != prev:
                transitions += 1
                prev = v

    for x in range(w):
        prev = 1 if pix[x] < mean else 0
        for y in range(1, h):
            v = 1 if pix[y * w + x] < mean else 0
            if v != prev:
                transitions += 1
                prev = v

    denom = (h * (w - 1)) + (w * (h - 1))
    transition_rate = transitions / denom if denom else 0
    if transition_rate < 0.04:
        return False

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_strength = (edge_stat.mean[0] or 0) / 255.0
    if edge_strength < 0.02:
        return False

    return True


def supabase_query(table, select_cols, filter_col, filter_val):
    """Execute a query against Supabase with proper error handling"""
    if not SUPABASE_URL or not ANON_KEY:
        logger.warning("Supabase credentials not configured")
        return None
    
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
        
        logger.debug(f"Querying {table} with {filter_col}={filter_val}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.debug(f"Found record in {table}")
                return data[0]
            else:
                logger.debug(f"No records found in {table} for {filter_col}={filter_val}")
        else:
            logger.error(f"Supabase query failed: {response.status_code} - {response.text}")
        
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout querying {table}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error querying {table}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying {table}: {str(e)}")
        return None


def init_supabase():
    """Initialize Supabase connection with robust error handling"""
    if not SUPABASE_URL or not ANON_KEY:
        logger.warning("Supabase credentials not available")
        return False
    
    try:
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {ANON_KEY}"
        }
        
        # Test with the fonts table instead of root endpoint
        logger.info("Testing Supabase connection...")
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/fonts?select=font_name&limit=1", 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Supabase connection successful")
            return True
        else:
            logger.error(f"Supabase connection failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Supabase connection timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Supabase connection error")
        return False
    except Exception as e:
        logger.error(f"Unexpected error initializing Supabase: {str(e)}")
        return False


supabase_available = init_supabase()

device = torch.device("cpu")
_MODEL = None
_CLASS_NAMES = None
_PREPROCESS = None
_MODEL_LOCK = Lock()


def get_model_bundle():
    global _MODEL, _CLASS_NAMES, _PREPROCESS
    if _MODEL is not None and _CLASS_NAMES is not None and _PREPROCESS is not None:
        return _MODEL, _CLASS_NAMES, _PREPROCESS

    with _MODEL_LOCK:
        if _MODEL is not None and _CLASS_NAMES is not None and _PREPROCESS is not None:
            return _MODEL, _CLASS_NAMES, _PREPROCESS

        # Load class names from labels.json to match training
        labels_path = os.path.join(ROOT_DIR, "training", "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            class_names = sorted(labels.keys(), key=lambda x: labels[x])
            logger.info(f"Loaded {len(class_names)} classes from labels.json")
        else:
            # Fallback to dataset folder structure
            dataset_path = os.path.join(ROOT_DIR, "dataset")
            class_names = sorted(entry.name for entry in os.scandir(dataset_path) if entry.is_dir())
            logger.warning(f"labels.json not found, using {len(class_names)} classes from dataset folder")
        
        num_classes = len(class_names)
        logger.info(f"Model expects {num_classes} classes")

        # Use weights parameter instead of pretrained
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            # Fallback for older torchvision versions
            model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Try loading from models/font_model.pth first, fallback to my_font_model.pth
        model_path = os.path.join(ROOT_DIR, "models", "font_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "my_font_model.pth")
            logger.warning(f"models/font_model.pth not found, trying fallback: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        logger.info(f"Model state dict loaded successfully")
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded and set to eval mode")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        _MODEL = model
        _CLASS_NAMES = class_names
        _PREPROCESS = preprocess
        return _MODEL, _CLASS_NAMES, _PREPROCESS

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
    row = supabase_query("fonts", "creator_name, social_link, status, purchase_link, image_url", "font_name", font_name)
    if not row:
        row = supabase_query("fonts", "creator_name, social_link, status, purchase_link, image_url", "name", font_name)
    return row if row else {}


@app.route('/predict', methods=['POST'])
def predict():
    logger.info("=== Prediction request started ===")
    if 'file' not in request.files:
        logger.error("Missing 'file' field in form data")
        return jsonify({'error': "Missing 'file' field in form data"}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename}, size: {file.content_length}")
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    try:
        logger.info("Opening image...")
        img = Image.open(file.stream)
        logger.info(f"Image opened successfully, mode: {img.mode}, size: {img.size}")
        img = ImageOps.exif_transpose(img)
        try:
            img.draft("L", (1024, 1024))
        except Exception as e:
            logger.warning(f"Image draft failed: {e}")
        img = img.convert("L")
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img.thumbnail((1024, 1024), resampling)
        logger.info(f"Image processed, final size: {img.size}")

        # Temporarily disable text validation to allow font recognition
        # if not looks_like_text(img):
        #     logger.info("Image does not look like text, returning Unknown")
        #     return jsonify({
        #         'font': 'Unknown',
        #         'confidence': 0.0,
        #         'creator_name': 'Unknown Creator',
        #         'creator': 'Unknown Creator',
        #         'social_link': '',
        #         'purchase_link': '',
        #         'status': 'unknown',
        #         'image_url': '',
        #         'type': 'free'
        #     })

        logger.info("Loading model bundle...")
        model, class_names, preprocess = get_model_bundle()
        logger.info(f"Model loaded with {len(class_names)} classes")
        logger.info(f"Class names: {class_names}")
        print(f"DEBUG: Class names length: {len(class_names)}")
        print(f"DEBUG: Class names: {class_names}")
        
        logger.info("Preprocessing image...")
        img_tensor = preprocess(img).unsqueeze(0)
        logger.info(f"Image tensor shape: {img_tensor.shape}")

        logger.info("Running inference...")
        with torch.inference_mode():
            outputs = model(img_tensor)
            logger.info(f"Model outputs shape: {outputs.shape}")
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_idx = pred.item()
            logger.info(f"Predicted index: {predicted_idx}, class_names length: {len(class_names)}")
            print(f"DEBUG: Predicted index: {predicted_idx}")
            print(f"DEBUG: Class names length: {len(class_names)}")
            if predicted_idx >= len(class_names):
                logger.error(f"Index out of range: predicted_idx={predicted_idx} >= len(class_names)={len(class_names)}")
                raise IndexError(f"Predicted index {predicted_idx} out of range for {len(class_names)} classes")
            predicted_class = class_names[predicted_idx]
            confidence = conf.item()
        logger.info(f"Prediction: {predicted_class}, confidence: {confidence:.4f}")
        print(f"DEBUG: Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        del img_tensor, outputs, probs, conf, pred

        if confidence < MIN_CONFIDENCE_FOR_MATCH:
            return jsonify({
                'font': 'Unknown',
                'confidence': confidence,
                'creator_name': 'Unknown Creator',
                'creator': 'Unknown Creator',
                'social_link': '',
                'purchase_link': '',
                'status': 'unknown',
                'image_url': '',
                'type': 'free'
            })

        creator_name = "Unknown Creator"
        social_link = ""
        status = "unknown"
        purchase_link = ""
        image_url = ""
        db_font_name = predicted_class

        if supabase_available:
            row = supabase_query("fonts", "font_name, creator_name, social_link, status, purchase_link, image_url", "font_name", predicted_class)
            if not row:
                row = supabase_query("fonts", "font_name, creator_name, social_link, status, purchase_link, image_url", "name", predicted_class)

            if row and isinstance(row, dict):
                db_font_name = row.get("font_name") or predicted_class
                creator_name = row.get("creator_name") or "Unknown Creator"
                social_link = row.get("social_link") or ""
                status = str(row.get("status") or "unknown")
                purchase_link = row.get("purchase_link") or ""
                image_url = row.get("image_url") or ""

        if creator_name == "Unknown Creator":
            fallback_data = MANUAL_FONT_FALLBACKS.get(predicted_class)
            if fallback_data:
                creator_name = fallback_data["creator_name"]
                social_link = fallback_data["social_link"]
                status = fallback_data["status"]
                purchase_link = fallback_data.get("purchase_link", "")

        status_normalized = status.lower()

        logger.info(f"Returning result: font={db_font_name}, confidence={confidence:.4f}")
        return jsonify({
            'font': db_font_name,
            'confidence': confidence,
            'creator_name': creator_name,
            'creator': creator_name,
            'social_link': social_link,
            'purchase_link': purchase_link,
            'status': status,
            'image_url': image_url,
            'type': 'premium' if status_normalized in ('premium', 'paid', 'pro') else 'free'
        })

    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Invalid image or prediction failed', 'details': str(e)}), 500


@app.route('/get_all_fonts', methods=['GET'])
def get_all_fonts():
    """Fetch fonts from Supabase with session-based pagination (6 fonts per page)"""
    if not supabase_available:
        logger.warning("Supabase not available, returning empty fonts list")
        return jsonify([])
    
    try:
        # Get page and seed parameters from query string
        page = int(request.args.get('page', 1))
        seed = request.args.get('seed', None)
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
        
        logger.info(f"Fetching all fonts from Supabase for page {page} with seed {seed}...")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            all_fonts = response.json()
            logger.info(f"Retrieved {len(all_fonts)} fonts from Supabase")
            
            if not all_fonts:
                logger.warning("No fonts found in database")
                return jsonify({
                    'fonts': [],
                    'currentPage': page,
                    'totalPages': 0,
                    'totalFonts': 0,
                    'seed': seed
                })
            
            # Use session-based seed for consistent random ordering
            if seed:
                try:
                    # Set the random seed for consistent ordering
                    random.seed(seed)
                    logger.info(f"Using session seed {seed} for consistent random ordering")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid seed {seed}, using default: {e}")
                    seed = None
            
            # Shuffle fonts with the seeded random generator
            random.shuffle(all_fonts)
            
            # Calculate pagination
            total_fonts = len(all_fonts)
            total_pages = (total_fonts + fonts_per_page - 1) // fonts_per_page
            
            # Get fonts for current page
            start_index = (page - 1) * fonts_per_page
            end_index = start_index + fonts_per_page
            page_fonts = all_fonts[start_index:end_index]
            
            logger.info(f"Returning {len(page_fonts)} fonts for page {page} of {total_pages}")
            return jsonify({
                'fonts': page_fonts,
                'currentPage': page,
                'totalPages': total_pages,
                'totalFonts': total_fonts,
                'seed': seed
            })
        else:
            logger.error(f"Failed to fetch fonts: {response.status_code}")
            return jsonify({
                'fonts': [],
                'currentPage': page,
                'totalPages': 0,
                'totalFonts': 0,
                'seed': seed
            })
    except Exception as e:
        logger.error(f"Error fetching fonts from Supabase: {str(e)}")
        return jsonify({
            'fonts': [],
            'currentPage': 1,
            'totalPages': 0,
            'totalFonts': 0,
            'seed': seed
        })


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(ROOT_DIR, 'ndex.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
