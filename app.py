"""
Gradio demo for multi-label emotion classification.

Loads the fine-tuned BERTweet model from the Hugging Face Hub and predicts
which emotions are present in a tweet. Designed to run on Hugging Face Spaces
(free CPU tier) but also works locally.

To run locally:
    pip install -r requirements-space.txt
    python app.py
"""

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The model repo on the Hugging Face Hub. Override with an env var if needed.
MODEL_ID = os.environ.get("MODEL_ID", "JaswantDev/bertweet-emotion-classification")
MAX_LENGTH = 96

# Load label names and per-label decision thresholds that ship inside the repo.
with open(hf_hub_download(MODEL_ID, "label_columns.json")) as f:
    label_columns = json.load(f)

with open(hf_hub_download(MODEL_ID, "thresholds.json")) as f:
    thresholds = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


def predict_emotions(text):
    """Return predicted emotions (above threshold) and all probabilities."""
    text = (text or "").strip()
    if not text:
        return {}, "Type a sentence above to see predictions."

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.sigmoid(logits).squeeze(0).numpy()

    # Build a sorted list of (label, probability) for every emotion.
    scored = sorted(
        ((label, float(probabilities[i])) for i, label in enumerate(label_columns)),
        key=lambda x: x[1],
        reverse=True,
    )

    # Emotions that cleared their tuned threshold = the actual prediction.
    predicted = {
        label: prob for label, prob in scored if prob >= thresholds[label]
    }

    # Top-k fallback: if nothing cleared a threshold, still show the top 3 so
    # the demo never looks empty.
    if predicted:
        label_output = predicted
    else:
        label_output = {label: prob for label, prob in scored[:3]}

    # A readable breakdown of every emotion's probability vs its threshold.
    lines = ["| Emotion | Probability | Threshold | Predicted |", "|---|---|---|---|"]
    for label, prob in scored:
        hit = "yes" if prob >= thresholds[label] else "—"
        lines.append(f"| {label} | {prob * 100:.1f}% | {thresholds[label]:.2f} | {hit} |")
    breakdown = "\n".join(lines)

    return label_output, breakdown


examples = [
    "I am excited but nervous about my interview tomorrow",
    "I can't believe they cancelled the show, this is so unfair",
    "Spending the whole day with my family made me so happy",
    "I'm scared this won't work out the way I hoped",
]

demo = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(
        lines=3,
        label="Tweet / sentence",
        placeholder="Type something with emotion...",
    ),
    outputs=[
        gr.Label(label="Predicted emotions"),
        gr.Markdown(label="Full breakdown"),
    ],
    examples=examples,
    title="Multi-Label Emotion Classifier (BERTweet)",
    description=(
        "Detects one or more emotions in a tweet using a BERTweet transformer "
        "fine-tuned on the SemEval-2018 Task 1 dataset, with per-label tuned "
        "thresholds. Multi-label: a sentence can carry several emotions at once."
    ),
    article=(
        "Best model: BERTweet + threshold tuning — Micro F1 0.7266, Macro F1 0.6187. "
        "Source code: https://github.com/Jaswant06/semeval-emotion-classification-pytorch"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
