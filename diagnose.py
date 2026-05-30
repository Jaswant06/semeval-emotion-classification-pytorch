"""
Calibration stress test for the BERTweet emotion classifier.

Feeds clearly-positive, clearly-negative, and neutral sentences through the
model to check whether its predictions are well-calibrated on plain English
(as opposed to the tweet-style text it was trained on). Reports the joy and
sadness probabilities plus the final thresholded prediction for each sentence.

Finding: the model is poorly calibrated on general English. It over-predicts
`sadness` on positive sentences (e.g. "I'm so proud of myself today" -> 89%
sadness), the tuned `joy` threshold (0.10) is low enough that neutral text is
tagged as joy, and several clearly-negative sentences predict nothing. See the
"Calibration stress test" section of the README for the full analysis.

Run:  python diagnose.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "models" / "bertweet_emotion_classifier"
MAX_LENGTH = 96

labels = json.load(open(MODEL_DIR / "label_columns.json"))
thresholds = json.load(open(MODEL_DIR / "thresholds.json"))
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

POSITIVE = [
    "I love this, best day ever!",
    "I am so happy and grateful right now",
    "This is wonderful news, I'm thrilled!",
    "Feeling amazing, everything is going great",
    "I'm so proud of myself today",
    "What a beautiful morning, I feel fantastic",
    "I can't stop smiling, this is the best",
    "So excited for my vacation next week!",
]
NEGATIVE = [
    "I hate this, it's the worst",
    "I'm so angry I could scream",
    "This is disgusting and makes me sick",
    "I feel hopeless and miserable",
    "I'm terrified of what comes next",
    "Everything is falling apart and I'm devastated",
]
NEUTRAL = [
    "The meeting is scheduled for 3pm tomorrow",
    "I bought some groceries on the way home",
    "The train arrives at the station every hour",
    "It is currently 20 degrees outside",
]


def probs(text):
    enc = tok(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        p = torch.sigmoid(model(**enc).logits).squeeze(0).numpy()
    return {labels[i]: float(p[i]) for i in range(len(labels))}


def predicted(pr):
    return [l for l in labels if pr[l] >= thresholds[l]]


def run(name, sentences):
    print(f"\n===== {name} =====")
    sad_pred = 0
    joy_pred = 0
    for s in sentences:
        pr = probs(s)
        pred = predicted(pr)
        if "sadness" in pred:
            sad_pred += 1
        if "joy" in pred:
            joy_pred += 1
        print(f"  joy={pr['joy']*100:5.1f}%  sad={pr['sadness']*100:5.1f}%  ->  {pred}")
        print(f"     \"{s}\"")
    print(f"  [{name}] sadness predicted in {sad_pred}/{len(sentences)} | joy predicted in {joy_pred}/{len(sentences)}")


run("CLEARLY POSITIVE", POSITIVE)
run("CLEARLY NEGATIVE", NEGATIVE)
run("NEUTRAL", NEUTRAL)
