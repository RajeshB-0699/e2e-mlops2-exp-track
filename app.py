import pickle
import os
import numpy as np
from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

MODEL_PATH = "mode/iris_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model file not found")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = [float(x) for x in requests.form.values()]
    prediction = model.predict([features])[0]
    return render_template("index.html",prediction_text = f"Predicted Iris class: {prediction}")

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5001)


