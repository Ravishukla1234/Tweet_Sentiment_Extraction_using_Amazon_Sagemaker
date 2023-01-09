# app.py
import os
import requests
import json
import re
from flask import Flask, render_template, request

app = Flask(__name__)


def predict_text(text, sentiment, url):
    request = {"inputs": text, "sentiment": sentiment}
    data = json.dumps(request)
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=data)
    print(response.text)
    prediction = json.loads(json.loads(response.text)["body"])
    print(f"prediction - {prediction}")
    output = prediction["answer"]
    return output


@app.route("/", methods=["GET", "POST"])
def predict():
    url = "<<YOUR API GATEWAY URL>>"
    if request.method == "POST":
        text = request.form.get("text")
        sentiment = request.form.get("sentiment")
        print(f"text - {text}")
        if text:
            output = predict_text(text, sentiment, url)
            return render_template(
                "index.html", prediction=output, text=text, sentiment=sentiment
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=12000, debug=True)
