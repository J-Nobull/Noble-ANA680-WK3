from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Manually assigned, rounded feature ranges
FEATURE_RANGES = {
    'fixed_acidity': (0, 17),
    'volatile_acidity': (0, 2),
    'citric_acid': (0, 2),
    'residual_sugar': (0, 70),
    'chlorides': (0, 1),
    'free_sulfur_dioxide': (0, 300),
    'total_sulfur_dioxide': (0, 500),
    'density': (0.95, 1.05),
    'pH': (2, 5),
    'sulphates': (0, 2),
    'alcohol': (7, 16),
}

FEATURE_KEYS = list(FEATURE_RANGES.keys())

with open("wine-qual.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_ranges=FEATURE_RANGES, prediction=None)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        features = []
        for key in FEATURE_KEYS:
            val = float(request.form[key])
            # Optionally clamp to allowed range
            min_val, max_val = FEATURE_RANGES[key]
            val = max(min_val, min(max_val, val))
            features.append(val)
        arr = np.array(features).reshape(1, -1)
        pred = model.predict(arr)[0]
        pred_display = f"Predicted wine quality: {pred:.2f}"
    except Exception as e:
        pred_display = f"Error: {e}"
    return render_template("index.html", feature_ranges=FEATURE_RANGES, prediction=pred_display)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
