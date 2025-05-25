from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
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
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
