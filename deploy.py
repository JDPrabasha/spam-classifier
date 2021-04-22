from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

df = pd.read_csv("ham_spam.csv", encoding="latin")
cleanup = {"v2": {"spam": 1, "ham": 0}}
df.replace(cleanup)

targets = df["v1"]
features = df["v2"]

vectorizer = CountVectorizer()
vectorizer.fit(features)
X = vectorizer.transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    X, targets, test_size=0.33, random_state=42)

model = GaussianNB().fit(X_train.toarray(), y_train)


def convert(text):
    return vectorizer.transform([text]).toarray()


def translate(text):
    if text == "spam":
        return "This is a spam email"
    else:
        return "This is not spam"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = request.form['emailtext']

    # final_features = [np.array(int_features)]
    prediction = model.predict(convert(int_features))

    output = prediction[0]

    return render_template('index.html', prediction_text=translate(output))


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="localhost")
