#Creating interactive app
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("C:/PythonProject1/.venv/Scripts/affiliation_model.pkl")
vectorizer = joblib.load("C:/PythonProject1/.venv/Scripts/vectorizer.pkl")

@app.route("/")
def home():
    return "Affiliation Predictor API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    name_input = data["Name:"]

    prediction = predict_affiliation(name_input)

    return jsonify({"Name": name, "Predicted affiliation": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)