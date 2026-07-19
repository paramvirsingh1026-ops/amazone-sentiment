import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

app = Flask(__name__)


# ---------- Data + training (runs once, at startup) ----------

def generate_synthetic_amazon_reviews(n_samples=600, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    products = [
        "wireless earbuds", "laptop stand", "kitchen blender", "yoga mat",
        "phone case", "bluetooth speaker", "office chair", "coffee maker",
        "running shoes", "backpack", "smartwatch", "desk lamp",
        "water bottle", "gaming mouse", "electric kettle", "camera tripod",
    ]
    positive_templates = [
        "I absolutely love this {p}, it works perfectly and exceeded my expectations.",
        "Great {p}! The quality is amazing and it arrived earlier than expected.",
        "This {p} is fantastic, best purchase I've made this year.",
        "Highly recommend this {p}, excellent build quality and very durable.",
        "The {p} works flawlessly, super happy with this purchase.",
        "Amazing value for money, this {p} is exactly what I needed.",
        "Very satisfied with the {p}, easy to use and looks great too.",
        "Five stars for this {p}, it's reliable and well made.",
    ]
    negative_templates = [
        "Terrible {p}, it stopped working after just two days.",
        "Very disappointed with this {p}, poor build quality and cheap materials.",
        "Waste of money, the {p} broke almost immediately.",
        "This {p} is awful, nothing like what was advertised.",
        "I regret buying this {p}, it does not work as described.",
        "Poor quality {p}, would not recommend to anyone.",
        "The {p} arrived damaged and customer service was unhelpful.",
        "Not worth the price, this {p} feels flimsy and unreliable.",
    ]
    neutral_templates = [
        "The {p} is okay, does the job but nothing special.",
        "Average {p}, works fine but the design could be better.",
        "It's a decent {p} for the price, though there are better options.",
        "The {p} is fine, meets basic expectations but nothing extraordinary.",
        "Not bad, not great — this {p} is just an average product.",
        "The {p} works as expected, no major complaints or praise.",
    ]

    rows = []
    for _ in range(n_samples):
        product = rng.choice(products)
        rating = rng.choice([1, 2, 3, 4, 5], p=[0.18, 0.12, 0.15, 0.25, 0.30])
        if rating <= 2:
            template = rng.choice(negative_templates)
        elif rating == 3:
            template = rng.choice(neutral_templates)
        else:
            template = rng.choice(positive_templates)
        text = template.format(p=product)
        if rng.random() < 0.3:
            extras = [
                " Shipping was fast.", " Packaging was good.",
                " I bought this for a gift.", " Would buy again.",
                " Delivery took a while.", " Setup was simple.",
            ]
            text += rng.choice(extras)
        rows.append({"product": product, "rating": int(rating), "review_text": text})

    df = pd.DataFrame(rows)

    def rating_to_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"

    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


print("Training sentiment model at startup...")
df = generate_synthetic_amazon_reviews(n_samples=600)
df["clean_text"] = df["review_text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"], test_size=0.2,
    random_state=RANDOM_STATE, stratify=df["sentiment"],
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_tfidf, y_train)
print("Model ready.")


def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    proba_dict = {cls: round(float(p), 3) for cls, p in zip(model.classes_, proba)}
    return prediction, proba_dict


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("review", "").strip()
    if not text:
        return render_template("index.html", error="Please enter a review.")
    prediction, proba = predict_sentiment(text)
    return render_template(
        "index.html", review=text, prediction=prediction, proba=proba
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("review", "").strip()
    if not text:
        return jsonify({"error": "Missing 'review' field"}), 400
    prediction, proba = predict_sentiment(text)
    return jsonify({"review": text, "sentiment": prediction, "probabilities": proba})


if __name__ == "__main__":
    app.run(debug=True)
