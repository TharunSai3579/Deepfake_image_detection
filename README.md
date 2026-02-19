# Deepfake_image_detection
Fake Image Detection Web App using Flask, MobileNetV2, and XGBoost. Upload an image to classify it as Real or Fake with confidence score and processing time.
An AI-powered web application that detects whether an uploaded image is Real or Fake using a hybrid deep learning + machine learning approach.

Built with Flask, MobileNetV2, and XGBoost, this system provides fast predictions along with confidence scores and processing time.

📌 Features

📤 Upload an image for prediction

🧠 Deep feature extraction using MobileNetV2

⚡ Fast classification using XGBoost

📊 Displays:

Prediction (Real / Fake)

Confidence score (%)

Processing time (ms)

🖼️ Image preview support

🧠 Model Architecture

Feature Extraction

Pretrained MobileNetV2 (ImageNet)

Converts image → feature vector

Classification

XGBoost model predicts Real/Fake

Preprocessing

Image resizing (128x128)

Normalization using scaler
