from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

FEATURE_KEYS = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Load LogReg model
with open("wine-qual.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = [float(data[key]) for key in FEATURE_KEYS]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify({"prediction": int(prediction[0])})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        features = [float(request.form[key]) for key in FEATURE_KEYS]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return render_template("index.html", prediction=int(prediction[0]))
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
