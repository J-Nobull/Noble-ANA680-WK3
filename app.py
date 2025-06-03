from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Define expected wine feature keys in order
FEATURE_KEYS = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Load XGBoost model
with open("xgb_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]

@app.route("/")
def home():
    return "Wine Quality Predictor API is now running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Ensure all required features are present
        features = [float(data[key]) for key in FEATURE_KEYS]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify({"prediction": int(prediction[0])})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
