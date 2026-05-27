# SemEval Emotion Classification With PyTorch

## Project Goal

This project builds a multi-label emotion classification system for tweets.

The goal is to classify one tweet into one or more emotions at the same time.

Example:

```text
Input:
I am excited but nervous about my interview tomorrow

Possible output:
anticipation
fear
optimism
```

This is different from basic sentiment analysis because a sentence can contain multiple emotions instead of only one label.

## Dataset

This project uses the SemEval-2018 Task 1 English emotion classification dataset.

Files used:

```text
data/2018-E-c-En-train.txt
data/2018-E-c-En-dev.txt
data/2018-E-c-En-test.txt
```

Dataset split:

```text
Training examples: 6,838
Development examples: 886
```

Emotion labels:

```text
anger
anticipation
disgust
fear
joy
love
optimism
pessimism
sadness
surprise
trust
```

Because this is a multi-label problem, one tweet can have several labels at once.

## Tools Used

- Python
- PyTorch
- scikit-learn
- pandas
- matplotlib
- seaborn
- Hugging Face Transformers
- BERTweet

## Models Built

This project compares three approaches.

### 1. TF-IDF + Logistic Regression

This is the classical machine learning baseline.

It uses:

```text
TF-IDF features
One-vs-Rest Logistic Regression
class_weight="balanced"
```

### 2. BiLSTM From Scratch

This model uses:

```text
tokenization
vocabulary
padding
embedding layer
bidirectional LSTM
dropout
multi-label output layer
```

The model is trained with:

```text
BCEWithLogitsLoss
sigmoid probabilities
per-label threshold tuning
```

### 3. BERTweet Transformer

This is the strongest model in the project.

BERTweet is a transformer model pretrained on English tweets, which makes it better suited for this dataset than a general English model.

It uses:

```text
vinai/bertweet-base
multi-label classification head
BCEWithLogitsLoss
per-label threshold tuning
```

## Results

Validation results on the development set:

| Model | Micro F1 | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.6051 | 0.5197 |
| BiLSTM | 0.5316 | 0.3630 |
| BiLSTM + Threshold Tuning | 0.5491 | 0.4541 |
| DistilBERT + Threshold Tuning | 0.7161 | 0.6116 |
| BERTweet + Threshold Tuning | 0.7266 | 0.6187 |

Best model:

```text
BERTweet + threshold tuning
```

Best performance:

```text
Micro F1: 0.7266
Macro F1: 0.6187
```

## BERTweet Classification Report

```text
              precision    recall  f1-score   support

       anger       0.76      0.85      0.81       315
anticipation       0.51      0.39      0.44       124
     disgust       0.73      0.86      0.79       319
        fear       0.77      0.79      0.78       121
         joy       0.83      0.89      0.86       400
        love       0.60      0.70      0.65       132
    optimism       0.71      0.83      0.77       307
   pessimism       0.38      0.51      0.43       100
     sadness       0.71      0.69      0.70       265
    surprise       0.38      0.34      0.36        35
       trust       0.21      0.28      0.24        43

   micro avg       0.69      0.76      0.73      2161
   macro avg       0.60      0.65      0.62      2161
weighted avg       0.70      0.76      0.73      2161
 samples avg       0.70      0.78      0.71      2161
```

## What I Learned

This project showed that stronger models are not always better automatically.

The classical TF-IDF baseline beat the first BiLSTM because the dataset is small and tweets are short. After adding threshold tuning, the BiLSTM improved, but the transformer model performed best.

Important lessons:

- Multi-label classification is different from multi-class classification.
- Each emotion needs its own sigmoid probability.
- Rare labels need different decision thresholds.
- Macro F1 is important because it shows performance on weaker/rarer emotions.
- Classical ML can be very strong on small text datasets.
- Transformers perform better when the pretrained model matches the domain.
- BERTweet works well here because the dataset contains tweets.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the baseline model:

```bash
python src/train_baseline.py
```

Train the BiLSTM model:

```bash
python src/train_bilstm.py
```

Train the transformer model:

```bash
python src/train_transformer.py
```

Run transformer predictions:

```bash
python src/predict_transformer.py
```

## Project Structure

```text
semeval-emotion-classification-pytorch/
+-- data/
|   +-- 2018-E-c-En-train.txt
|   +-- 2018-E-c-En-dev.txt
|   +-- 2018-E-c-En-test.txt
+-- models/
|   +-- bilstm_emotion_classifier.pth
|   +-- vocab.json
|   +-- label_columns.json
|   +-- thresholds.json
|   +-- bertweet_emotion_classifier/
+-- plots/
|   +-- label_distribution.png
|   +-- baseline_confusion_matrices.png
|   +-- bilstm_training_curves.png
|   +-- transformer_training_curves.png
+-- src/
|   +-- train_baseline.py
|   +-- train_bilstm.py
|   +-- train_transformer.py
|   +-- predict_transformer.py
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Limitations

The model performs well on validation data, but real-world emotion prediction is still difficult.

Current limitations:

- Some rare labels are harder to predict, especially `trust` and `surprise`.
- The model can miss subtle mixed emotions.
- Thresholds are optimized for F1 score, not always human intuition.
- Tweets are noisy, short, and context-dependent.
- Sarcasm and political statements are still difficult.

## Next Improvements

Possible future upgrades:

- Add class imbalance handling with `pos_weight`
- Tune thresholds on a separate validation set
- Add early stopping
- Compare BERTweet with RoBERTa and DeBERTa
- Build a small Streamlit or FastAPI demo
- Deploy the best model as an API
- Add a model card with intended use and limitations

## Final Takeaway

This project demonstrates a complete NLP modeling workflow:

```text
classical baseline -> neural sequence model -> transformer fine-tuning -> model comparison
```

The strongest model was:

```text
BERTweet + threshold tuning
```

This is the model used for final prediction.
