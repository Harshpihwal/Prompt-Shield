# Methodology

## Overview

PromptShield employs **supervised multi-class text classification** using a hybrid stacked ensemble approach. The methodology combines lexical feature extraction (TF-IDF), classical machine learning (Random Forest, XGBoost), and transformer-based deep learning (RoBERTa) to classify prompts into five threat categories.

---

## 1. Data Loading

The dataset is loaded from an Excel file containing two columns:
- `prompt` — the input text
- `label` — integer class label (0–4)

```python
df = pd.read_excel('Prompt_Shield_Fixed_v3.xlsx')
```

---

## 2. Text Preprocessing

All prompts undergo the following cleaning steps:

1. **Lowercase conversion** — standardizes case
2. **URL removal** — strips `http://` and `https://` links
3. **Special character removal** — keeps only alphanumeric characters and spaces
4. **Whitespace normalization** — collapses multiple spaces into one
5. **Deduplication** — removes exact duplicate prompts
6. **Drop NaN** — removes any null entries

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## 3. Data Splitting

An **80/20 stratified train/test split** ensures proportional class representation:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Resulting split**: Train = 4,808 samples, Test = 1,202 samples

---

## 4. Data Augmentation (Training Only)

Three augmentation techniques are applied to each training sample, producing 4× the original training set:

### Synonym Replacement
Replaces a random word with its WordNet synonym.

### Random Deletion
Removes words with probability p=0.15.

### Random Swap
Swaps two randomly selected words in the sentence.

> **Important**: Augmentation is applied **only during training**, never to validation or test data. This prevents augmented near-duplicates from leaking into evaluation sets.

**Augmented training size**: 19,232 samples (from 4,808 original)

---

## 5. Feature Extraction

### TF-IDF Vectorization

Converts text into sparse numerical vectors using Term Frequency–Inverse Document Frequency:

```python
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3
)
```

- **Max features**: 3,000 — limits dimensionality
- **N-gram range**: (1, 2) — captures unigrams and bigrams
- **Sublinear TF**: Applies logarithmic scaling to term frequency
- **Min document frequency**: 3 — filters rare terms

### RoBERTa Tokenization

For the transformer model, raw text is tokenized using the RoBERTa tokenizer:

```python
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Max sequence length: 128 tokens
```

---

## 6. Base Model Training

### Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=6,
    max_features='sqrt',
    random_state=42
)
```

- Trains on TF-IDF features
- Multiple decision trees analyze different feature subsets
- Final prediction based on majority vote across trees

### XGBoost Classifier

```python
XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=2.0,
    reg_lambda=3.0,
    min_child_weight=5,
    eval_metric='mlogloss',
    random_state=42
)
```

- Trains on TF-IDF features
- Sequential boosting corrects previous errors
- Strong L1/L2 regularization prevents overfitting

### RoBERTa Classifier

```python
RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=5,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)
```

- Fine-tuned for 3 epochs on augmented training data
- Learning rate: 2e-5 with AdamW optimizer (weight decay 0.01)
- Gradient clipping at norm 1.0
- Captures contextual and semantic relationships

---

## 7. Stacked Ensemble (Out-of-Fold)

The stacking process uses **5-fold Stratified Cross-Validation** to prevent data leakage:

### Process

For each of the 5 folds:
1. Split training data into fold-train and fold-validation
2. Apply augmentation to fold-train only
3. Train TF-IDF vectorizer on fold-train
4. Train Random Forest on fold-train TF-IDF features
5. Train XGBoost on fold-train TF-IDF features
6. Train a fresh RoBERTa model on fold-train (2 epochs)
7. Collect probability predictions on fold-validation from all 3 models
8. Store as out-of-fold (OOF) meta-features

### Meta-Feature Construction

For each sample, the meta-feature vector is:

```
[RF_p0, RF_p1, RF_p2, RF_p3, RF_p4,    # 5 RF probabilities
 XGB_p0, XGB_p1, XGB_p2, XGB_p3, XGB_p4, # 5 XGBoost probabilities
 ROB_p0, ROB_p1, ROB_p2, ROB_p3, ROB_p4]  # 5 RoBERTa probabilities
```

**Total OOF meta-feature shape**: (4,808, 15)

### Meta-Classifier

```python
meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
meta_model.fit(oof_meta, oof_labels)
```

The Logistic Regression meta-classifier learns the optimal weighting of each base model's predictions.

---

## 8. Inference Pipeline

### Step 1: Harmful Keyword Check
A first-pass keyword filter checks against a curated list of harmful phrases. If matched, immediately returns label 4 (Harmful) with 99% confidence.

### Step 2: Spell Correction
PySpellChecker corrects misspelled words while skipping technical terms, city names, food terms, and other domain-specific vocabulary (134 skip words).

### Step 3: Text Cleaning
Standard preprocessing (same as training).

### Step 4: Model Ensemble
Run cleaned text through all three base models and collect probability distributions.

### Step 5: Meta-Classification
Concatenate base model probabilities and pass through the meta-classifier.

### Step 6: Confidence Thresholding
- If confidence ≥ 60%: return predicted label with confidence
- If confidence < 60%: flag as UNCERTAIN, provide top-2 predictions, recommend manual review

---

## 9. Evaluation Strategy

| Method | Purpose |
|--------|---------|
| **Holdout test set** (20%) | Final model performance assessment |
| **5-Fold K-Fold CV** | General cross-validation on meta-features |
| **5-Fold Stratified K-Fold CV** | Class-balanced cross-validation |
| **Repeated K-Fold (5×2)** | Stability assessment |
| **Overfitting check** | Compare meta-train vs meta-test accuracy |

### Metrics Used
- Accuracy
- Precision (per class)
- Recall (per class)
- F1-Score (per class)
- Confusion matrix
- Per-class accuracy
