# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:05:50 2026

@author: Dell
"""


from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load trained model
model = joblib.load("sentiment_model.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    review = ""

    if request.method == "POST":

        review = request.form["review"]

        cleaned_review = clean_text(review)

        prediction = model.predict([cleaned_review])[0]

        if prediction == "positive":
            result = "Positive Review 😊"
        else:
            result = "Negative Review 😞"

    return render_template(
        "index.html",
        result=result,
        review=review
    )

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
