from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
from feature_extract import extract_features

app = Flask(__name__)
CORS(app)

# Load trained SVM model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image is uploaded
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        save_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(save_path)

        # Extract features from uploaded image
        features = extract_features(save_path)

        # Predict using SVM model
        prediction = model.predict([features])[0]

        # Convert numeric prediction → label
        result = "real" if prediction == 0 else "ai"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "AI Image Detector Backend Running!"


if __name__ == "__main__":
    app.run(debug=True)
