import os
import sys

# Ensure project root is on the path so model files are found
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import xgboost as xgb
import time
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import io

BUILD_DIR = os.path.join(ROOT_DIR, "build")

app = Flask(__name__)

# Load model & scaler parameters without pulling in scikit-learn at runtime
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(ROOT_DIR, "xgb_model.json"))
_scaler_params = np.load(os.path.join(ROOT_DIR, "scaler_params.npz"))
_scaler_mean = _scaler_params["mean"]
_scaler_scale = _scaler_params["scale"]

# Feature extractor (TFLite)
IMG_SIZE = 128
interpreter = Interpreter(model_path=os.path.join(ROOT_DIR, "mobilenetv2_feature_extractor.tflite"))
interpreter.allocate_tensors()
_input_details = interpreter.get_input_details()
_output_details = interpreter.get_output_details()

def extract_features(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(_input_details[0]["index"], image)
    interpreter.invoke()
    return interpreter.get_tensor(_output_details[0]["index"])


def scale_features(features):
    return (features - _scaler_mean) / _scaler_scale


def predict_proba(features):
    # Booster.predict returns positive-class probability for binary models.
    pos_prob = float(xgb_model.predict(xgb.DMatrix(features))[0])
    return np.array([1.0 - pos_prob, pos_prob], dtype=np.float32)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the React/static build folder. Fall back to index.html for SPA routing."""
    if path and os.path.exists(os.path.join(BUILD_DIR, path)):
        return send_from_directory(BUILD_DIR, path)
    return send_from_directory(BUILD_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    start = time.time()

    file = request.files["image"]
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    features = extract_features(image)
    features = scale_features(features)

    prob = predict_proba(features)
    prediction = int(np.argmax(prob))

    result = "Fake" if prediction == 1 else "Real"
    confidence = round(float(prob[prediction]) * 100, 2)
    processing_time = round((time.time() - start) * 1000, 2)

    return jsonify({
        "result": result,
        "confidence": confidence,
        "processing_time": processing_time,
    })

if __name__ == "__main__":
    app.run(debug=True)
