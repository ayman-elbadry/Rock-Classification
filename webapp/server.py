import os
import sys
import json
import base64
import io
import glob
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from explicability import get_gradcam

app = Flask(__name__, static_folder='static')

# ─── Configuration ───────────────────────────────────────────────
CLASSES = ['Igneous', 'Metamorphic', 'Sedimentary']
SUBTYPES = {
    'Igneous': ['Basalt', 'Granite'],
    'Metamorphic': ['Marble', 'Quartzite'],
    'Sedimentary': ['Coal', 'Limestone', 'Sandstone']
}
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_georock_model.pth')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'Dataset')
TRAINING_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'training_results')
EDA_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'eda_results')

model = None
device = None


def load_model():
    global model, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[OK] Model loaded from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {MODEL_PATH}")

    model = model.to(device)
    model.eval()


def pil_to_base64(img):
    """Convert a PIL image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ─── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Classify an uploaded rock image and return Grad-CAM overlay."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    try:
        overlay_img, class_idx, probs = get_gradcam(model, image)

        result = {
            'prediction': CLASSES[class_idx],
            'confidence': float(probs[class_idx]) * 100,
            'probabilities': {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASSES))},
            'gradcam': pil_to_base64(overlay_img),
            'subtypes': SUBTYPES.get(CLASSES[class_idx], [])
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """Return model status and available training results."""
    model_exists = os.path.exists(MODEL_PATH)

    training_images = []
    if os.path.isdir(TRAINING_RESULTS_DIR):
        for fname in sorted(os.listdir(TRAINING_RESULTS_DIR)):
            if fname.endswith('.png'):
                training_images.append(fname)

    eda_images = []
    if os.path.isdir(EDA_RESULTS_DIR):
        for fname in sorted(os.listdir(EDA_RESULTS_DIR)):
            if fname.endswith('.png'):
                eda_images.append(fname)

    # Read classification report text if available
    report_text = None
    report_path = os.path.join(TRAINING_RESULTS_DIR, 'classification_report.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_text = f.read()

    return jsonify({
        'model_loaded': model_exists,
        'device': str(device),
        'classes': CLASSES,
        'subtypes': SUBTYPES,
        'training_images': training_images,
        'eda_images': eda_images,
        'classification_report': report_text
    })


@app.route('/api/training-image/<filename>')
def training_image(filename):
    """Serve training result images."""
    return send_from_directory(os.path.abspath(TRAINING_RESULTS_DIR), filename)


@app.route('/api/eda-image/<filename>')
def eda_image(filename):
    """Serve EDA result images."""
    return send_from_directory(os.path.abspath(EDA_RESULTS_DIR), filename)


@app.route('/api/sample-images')
def sample_images():
    """Return sample images from the dataset for the gallery."""
    samples = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for subtype in os.listdir(class_dir):
            subtype_dir = os.path.join(class_dir, subtype)
            if not os.path.isdir(subtype_dir):
                continue
            images = [f for f in os.listdir(subtype_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                # Pick up to 2 random samples per subtype
                chosen = random.sample(images, min(2, len(images)))
                for img_name in chosen:
                    img_path = os.path.join(subtype_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img.thumbnail((300, 300))
                    samples.append({
                        'class': class_name,
                        'subtype': subtype,
                        'image': pil_to_base64(img),
                        'filename': img_name
                    })
    return jsonify(samples)


if __name__ == '__main__':
    load_model()
    app.run(debug=False, port=5000, host='0.0.0.0')
