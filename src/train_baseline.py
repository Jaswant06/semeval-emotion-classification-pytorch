from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "2018-E-c-En-train.txt"
DEV_PATH = DATA_DIR / "2018-E-c-En-dev.txt"

train_df = pd.read_csv(TRAIN_PATH, sep="\t")
dev_df = pd.read_csv(DEV_PATH, sep="\t")

print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)
print("Columns:")
print(train_df.columns.tolist())

text_column = "Tweet"
label_columns = [
    col for col in train_df.columns
    if col not in ["ID", "Tweet"]
]

print("Label columns:", label_columns)

X_train_text = train_df[text_column].astype(str)
X_dev_text = dev_df[text_column].astype(str)

y_train = train_df[label_columns]
y_dev = dev_df[label_columns]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
)

X_train = vectorizer.fit_transform(X_train_text)
X_dev = vectorizer.transform(X_dev_text)

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
)

model.fit(X_train, y_train)

dev_predictions = model.predict(X_dev)

micro_f1 = f1_score(y_dev, dev_predictions, average="micro")
macro_f1 = f1_score(y_dev, dev_predictions, average="macro")

print("\nBaseline Results")
print("Micro F1:", round(micro_f1, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_dev,
        dev_predictions,
        target_names=label_columns,
        zero_division=0,
    )
)

label_counts = y_train.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=label_counts.index,
    y=label_counts.values,
)
plt.title("Training Label Counts")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "label_distribution.png", bbox_inches="tight")
plt.close()

conf_matrices = multilabel_confusion_matrix(y_dev, dev_predictions)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for index, label in enumerate(label_columns):
    sns.heatmap(
        conf_matrices[index],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[index],
        cbar=False,
    )
    axes[index].set_title(label)
    axes[index].set_xlabel("Predicted")
    axes[index].set_ylabel("Actual")

for index in range(len(label_columns), len(axes)):
    axes[index].axis("off")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "baseline_confusion_matrices.png", bbox_inches="tight")
plt.close()

print("\nSaved plots:")
print("plots/label_distribution.png")
print("plots/baseline_confusion_matrices.png")