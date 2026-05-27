from pathlib import Path
import re
import json
from collections import Counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "2018-E-c-En-train.txt"
DEV_PATH = DATA_DIR / "2018-E-c-En-dev.txt"

MAX_LEN = 50
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
MIN_WORD_FREQ = 2


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9'!?]+", " ", text)
    return text.split()


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


train_df = pd.read_csv(TRAIN_PATH, sep="\t")
dev_df = pd.read_csv(DEV_PATH, sep="\t")

TEXT_COLUMN = "Tweet"
LABEL_COLUMNS = [
    col for col in train_df.columns
    if col not in ["ID", "Tweet"]
]

print("Labels:", LABEL_COLUMNS)
print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)

counter = Counter()

for text in train_df[TEXT_COLUMN]:
    counter.update(tokenize(text))

vocab = {"<PAD>": 0, "<UNK>": 1}

for word, count in counter.items():
    if count >= MIN_WORD_FREQ:
        vocab[word] = len(vocab)

print("Vocab size:", len(vocab))


def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids


class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe[TEXT_COLUMN].astype(str).tolist()
        self.labels = dataframe[LABEL_COLUMNS].values.astype("float32")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        x = torch.tensor(encode(self.texts[index]), dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


train_dataset = EmotionDataset(train_df)
dev_dataset = EmotionDataset(dev_df)

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


class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(embedded)

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (hidden_forward, hidden_backward),
            dim=1,
        )

        hidden_combined = self.dropout(hidden_combined)
        logits = self.fc(hidden_combined)

        return logits


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

model = BiLSTMEmotionClassifier(
    vocab_size=len(vocab),
    num_labels=len(LABEL_COLUMNS),
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        logits = model(texts)
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
        for texts, labels in dev_loader:
            texts = texts.to(device)

            logits = model(texts)
            probs = torch.sigmoid(logits).cpu()

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

        torch.save(
            model.state_dict(),
            MODELS_DIR / "bilstm_emotion_classifier.pth",
        )

        with open(MODELS_DIR / "vocab.json", "w") as f:
            json.dump(vocab, f)

        with open(MODELS_DIR / "label_columns.json", "w") as f:
            json.dump(LABEL_COLUMNS, f)

        with open(MODELS_DIR / "thresholds.json", "w") as f:
            json.dump(best_thresholds, f)

        print("Saved new best model.")

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
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS + 1), micro_f1_scores, marker="o")
plt.title("Tuned Micro F1")
plt.xlabel("Epoch")
plt.ylabel("F1")

plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS + 1), macro_f1_scores, marker="o")
plt.title("Tuned Macro F1")
plt.xlabel("Epoch")
plt.ylabel("F1")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bilstm_training_curves.png", bbox_inches="tight")
plt.close()

print("\nSaved:")
print("models/bilstm_emotion_classifier.pth")
print("models/vocab.json")
print("models/label_columns.json")
print("models/thresholds.json")
print("plots/bilstm_training_curves.png")