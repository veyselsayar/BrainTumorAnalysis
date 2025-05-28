import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import sys

# add_result fonksiyonunu import et
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))
from ai.add_result import add_result

MODEL_PATH = "/Users/veysel/Desktop/DerinOgrenme/brain_tumor_optimized_cnn_model.keras"
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

app = Flask(__name__)
IMG_SIZE = 150

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    img_bytes = file.read()
    try:
        img_array = prepare_image(img_bytes)
        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(np.max(preds))
        results = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": [float(p) for p in preds[0]],
            "timestamp": time.time()
        }
        # Havuz dosyasına ekle
        add_result(results)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Brain Tumor CNN API is running!"

if __name__ == '__main__':
    model = tf.keras.models.load_model(MODEL_PATH)
    app.run(host="0.0.0.0", port=500, debug=True)
