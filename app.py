from flask import Flask, render_template, request
import numpy as np
import joblib
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import io

app = Flask(__name__)

# Load model & scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature extractor
IMG_SIZE = 128
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

def extract_features(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return feature_extractor.predict(image)

@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = processing_time = image_data = None

    if request.method == "POST":
        start = time.time()

        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        # Convert image for display
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