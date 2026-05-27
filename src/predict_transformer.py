from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "bertweet_emotion_classifier"

LABELS_PATH = MODEL_DIR / "label_columns.json"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"


with open(LABELS_PATH, "r") as f:
    label_columns = json.load(f)

with open(THRESHOLDS_PATH, "r") as f:
    thresholds = json.load(f)


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def predict_emotions(text):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

    results = []

    for index, label in enumerate(label_columns):
        probability = float(probabilities[index])
        threshold = thresholds[label]
        predicted = probability >= threshold

        results.append(
            {
                "emotion": label,
                "probability": probability,
                "threshold": threshold,
                "predicted": predicted,
            }
        )

    results = sorted(
        results,
        key=lambda item: item["probability"],
        reverse=True,
    )

    return results


print("Multi-Label Emotion Predictor")
print("Type a sentence. Type 'quit' to stop.\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "quit":
        break

    predictions = predict_emotions(text)

    predicted_emotions = [
        item for item in predictions
        if item["predicted"]
    ]

    print("\nPredicted emotions:")

    if predicted_emotions:
        for item in predicted_emotions:
            print(
                f"- {item['emotion']}: "
                f"{item['probability'] * 100:.2f}% "
                f"(threshold {item['threshold']:.2f})"
            )
    else:
        print("- none above threshold")

    print("\nTop probabilities:")
    for item in predictions[:5]:
        print(
            f"- {item['emotion']}: "
            f"{item['probability'] * 100:.2f}%"
        )

    print()