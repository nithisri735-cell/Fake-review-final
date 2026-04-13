from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "dl_artifacts" / "best_model"

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

LABEL_MAP = {
    0: "real",
    1: "fake",
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    review_text = str(payload.get("review", "")).strip()

    if not review_text:
        return jsonify({"error": "Please enter or speak a review before predicting."}), 400

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][pred_id].item())

    return jsonify(
        {
            "review": review_text,
            "prediction": pred_id,
            "label": LABEL_MAP[pred_id],
            "confidence": round(confidence, 4),
            "message": f"The review is predicted as {LABEL_MAP[pred_id].upper()}",
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
