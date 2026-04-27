import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    hrs_Studied = float(request.form["hrs_Studied"])
    prev_score = float(request.form["prev_score"])
    sleep = int(request.form["sleep"])
    sample_practiced = int(request.form["sample_practiced"])
    extra_Curr = 1 if request.form["extra_Curr"] == "Yes" else 0

    # Prepare features for prediction
   
    feature_names = ["hrs_Studied", "prev_score", "sleep", "sample_practiced", "extra_Curr"]
    features = pd.DataFrame([[hrs_Studied, prev_score, sleep, sample_practiced, extra_Curr]], columns=feature_names)

    # Predict charges
    prediction = model.predict(features)
    #  Format to float and 2 decimal places
    formatted_prediction = f"The predicted value is ${round(float(prediction), 2)}"


    return render_template("result.html", prediction=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
