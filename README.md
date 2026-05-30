# SemEval Emotion Classification With PyTorch

**🤗 Live demo:** [Try the model on Hugging Face Spaces](https://huggingface.co/spaces/JaswantDev/tweet-emotion-classifier)
&nbsp;|&nbsp; **Model:** [JaswantDev/bertweet-emotion-classification](https://huggingface.co/JaswantDev/bertweet-emotion-classification)

Type any tweet or sentence into the demo and the model predicts which emotions it expresses (multi-label).

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

Run transformer predictions in the terminal:

```bash
python src/predict_transformer.py
```

Run the interactive web demo locally (downloads the trained model from the Hugging Face Hub automatically):

```bash
pip install -r requirements-space.txt
python app.py
```

> Note: the trained model weights are not stored in this repo (they are large). The best model is hosted on the Hugging Face Hub at
> [JaswantDev/bertweet-emotion-classification](https://huggingface.co/JaswantDev/bertweet-emotion-classification) and is loaded automatically by `app.py`.
> To reproduce the weights locally instead, run the training scripts above.

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
+-- app.py                     # Gradio web demo (deployed on Hugging Face Spaces)
+-- diagnose.py                # calibration stress test (see Limitations)
+-- requirements.txt           # dependencies for training
+-- requirements-space.txt     # dependencies for the web demo
+-- README.md
+-- .gitignore
```

> Trained model weights (`models/`) are gitignored because of their size. See the
> [How To Run](#how-to-run) note for how the demo loads the model from the Hub.

## Limitations

The model performs well on validation data, but real-world emotion prediction is still difficult.

Current limitations:

- Some rare labels are harder to predict, especially `trust` and `surprise`.
- The model can miss subtle mixed emotions.
- Thresholds are optimized for F1 score, not always human intuition.
- Tweets are noisy, short, and context-dependent.
- Sarcasm and political statements are still difficult.

### Out-of-distribution text and the label-taxonomy ceiling

This is the most important limitation to understand. The model was trained on
short, reactive **tweets** (SemEval-2018). When given longer, reflective text —
for example a diary-style passage — it tends to predict confidently and
**incorrectly**, because such text is *out of distribution*: nothing like it
appears in the training data, so the model pattern-matches on surface features
rather than genuine emotional meaning.

A real example tested on the demo: a calm, melancholic personal reflection
(sitting alone on a rainy day, noticing a one-eyed pigeon that didn't fly away)
was predicted as `disgust` and `anger` — emotions the writer did not feel at
all. Two separate failures are at work:

1. **Domain shift.** Reflective prose is structurally and stylistically unlike
   tweets, so the learned patterns do not transfer.
2. **Label-taxonomy ceiling.** The 11 SemEval emotions cannot represent feelings
   such as *loneliness, melancholy, tenderness, or compassion*. Even a perfect
   classifier limited to this taxonomy could not label them correctly — the
   closest available label (`sadness`) scored just below its threshold here.
3. **Truncation.** Inputs are cut to 96 tokens, so long passages are only
   partially read.

The takeaway: a model's reliability is bounded by the **distribution it was
trained on** and the **label set it was given**. For genuinely open-ended human
emotion, a model trained on a richer taxonomy (e.g. the 27-label GoEmotions
dataset) would be a more appropriate starting point than this tweet-specific
model.

### Calibration stress test

To check whether the failures above were just "hard inputs" or a deeper problem,
I stress-tested the model on clearly-positive, clearly-negative, and neutral
sentences (see [diagnose.py](diagnose.py)). The model is **poorly calibrated on
plain English**:

- **Over-predicts `sadness` on positive text.** `"I'm so proud of myself today"`
  → 2% joy, **89% sadness**, predicted *disgust + pessimism + sadness*.
  `"I am so happy and grateful right now"` → 80% sadness.
- **The tuned `joy` threshold (0.10) is too low**, so neutral text gets tagged as
  joy: `"The train arrives every hour"` → predicted *joy*.
- **Under-predicts on negatives:** `"I feel hopeless and miserable"` → predicted
  *nothing*.

There are two distinct causes, and separating them matters:

1. **Threshold miscalibration (cheap to fix).** Thresholds were tuned to maximize
   F1 on the dev set, producing extreme values (e.g. `joy = 0.10`). Resetting them
   to saner values would remove the neutral-text noise.
2. **Weak underlying model on non-tweet English (not cheap to fix).** The *raw*
   sadness probability for "proud of myself" is 89% — no threshold change fixes
   that. The model genuinely struggles on text unlike its training tweets, which
   would require retraining or a different base model/dataset.

**Conclusion:** this model is reasonable on short, tweet-like text but should not
be treated as a general-purpose emotion classifier. Rather than hide that, it is
documented here, because honestly characterizing where a model breaks is part of
deploying it responsibly.

## Deployment

The best model (BERTweet + threshold tuning) is deployed two ways:

- **Model weights** are hosted on the Hugging Face Hub: [JaswantDev/bertweet-emotion-classification](https://huggingface.co/JaswantDev/bertweet-emotion-classification)
- **Interactive demo** runs on Hugging Face Spaces (Gradio): [tweet-emotion-classifier](https://huggingface.co/spaces/JaswantDev/tweet-emotion-classifier)

The demo loads the model from the Hub, applies per-label tuned thresholds, and shows both the predicted emotions and the full probability breakdown. See [app.py](app.py).

## Next Improvements

Possible future upgrades:

- Add class imbalance handling with `pos_weight`
- Tune thresholds on a separate validation set
- Add early stopping
- Compare BERTweet with RoBERTa and DeBERTa
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
