import os
import sys

# Ensure project root is on the path so model files are found
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from flask import Flask, render_template, request
import numpy as np
import joblib
import time
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import base64
import io

app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, "templates"))

# Load model & scaler
xgb_model = joblib.load(os.path.join(ROOT_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(ROOT_DIR, "scaler.pkl"))

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

@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = processing_time = image_data = None

    if request.method == "POST":
        start = time.time()

        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

        features = extract_features(image)
        features = scaler.transform(features)

        prob = xgb_model.predict_proba(features)[0]
        prediction = np.argmax(prob)

        result = "Fake" if prediction == 1 else "Real"
        confidence = round(float(prob[prediction]) * 100, 2)
        processing_time = round((time.time() - start) * 1000, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        processing_time=processing_time,
        image_data=image_data
    )

if __name__ == "__main__":
    app.run(debug=True)
