from pathlib import Path
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "2018-E-c-En-train.txt"
DEV_PATH = DATA_DIR / "2018-E-c-En-dev.txt"

MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 96
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5

train_df = pd.read_csv(TRAIN_PATH, sep="\t")
dev_df = pd.read_csv(DEV_PATH, sep="\t")

TEXT_COLUMN = "Tweet"
LABEL_COLUMNS = [
    col for col in train_df.columns
    if col not in ["ID", "Tweet"]
]

NUM_LABELS = len(LABEL_COLUMNS)

print("Model:", MODEL_NAME)
print("Labels:", LABEL_COLUMNS)
print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)


def tune_thresholds(y_true, probabilities, label_columns):
    best_thresholds = {}

    for index, label in enumerate(label_columns):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in [x / 100 for x in range(10, 91, 5)]:
            preds = (probabilities[:, index] >= threshold).astype(int)
            score = f1_score(
                y_true[:, index],
                preds,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        best_thresholds[label] = best_threshold

    return best_thresholds


def apply_thresholds(probabilities, thresholds, label_columns):
    predictions = []

    for row in probabilities:
        row_predictions = []

        for index, label in enumerate(label_columns):
            threshold = thresholds[label]
            row_predictions.append(int(row[index] >= threshold))

        predictions.append(row_predictions)

    return predictions


class EmotionTransformerDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe[TEXT_COLUMN].astype(str).tolist()
        self.labels = dataframe[LABEL_COLUMNS].values.astype("float32")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
        }

        return item


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EmotionTransformerDataset(train_df, tokenizer)
dev_dataset = EmotionTransformerDataset(dev_df, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_losses = []
micro_f1_scores = []
macro_f1_scores = []

best_macro_f1 = 0
best_thresholds = None
best_predictions = None
best_labels = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.sigmoid(outputs.logits).cpu()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    thresholds = tune_thresholds(
        y_true=all_labels,
        probabilities=all_probs,
        label_columns=LABEL_COLUMNS,
    )

    predictions = apply_thresholds(
        probabilities=all_probs,
        thresholds=thresholds,
        label_columns=LABEL_COLUMNS,
    )

    micro_f1 = f1_score(
        all_labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        all_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1_scores.append(micro_f1)
    macro_f1_scores.append(macro_f1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Loss: {average_loss:.4f} - "
        f"Tuned Micro F1: {micro_f1:.4f} - "
        f"Tuned Macro F1: {macro_f1:.4f}"
    )

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_thresholds = thresholds
        best_predictions = predictions
        best_labels = all_labels

        save_dir = MODELS_DIR / "bertweet_emotion_classifier"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        with open(save_dir / "label_columns.json", "w") as f:
            json.dump(LABEL_COLUMNS, f)

        with open(save_dir / "thresholds.json", "w") as f:
            json.dump(best_thresholds, f)

        print("Saved new best transformer model.")

print("\nBest Tuned Macro F1:", round(best_macro_f1, 4))

print("\nBest thresholds:")
for label, threshold in best_thresholds.items():
    print(f"{label}: {threshold}")

print("\nBest Classification Report:")
print(
    classification_report(
        best_labels,
        best_predictions,
        target_names=LABEL_COLUMNS,
        zero_division=0,
    )
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.title("Transformer Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS + 1), micro_f1_scores, marker="o")
plt.title("Transformer Tuned Micro F1")
plt.xlabel("Epoch")
plt.ylabel("F1")

plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS + 1), macro_f1_scores, marker="o")
plt.title("Transformer Tuned Macro F1")
plt.xlabel("Epoch")
plt.ylabel("F1")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "transformer_training_curves.png", bbox_inches="tight")
plt.close()

print("\nSaved:")
print("models/bertweet_emotion_classifier/")
print("plots/transformer_training_curves.png")