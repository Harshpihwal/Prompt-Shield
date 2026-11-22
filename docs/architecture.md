# System Architecture

## Overview

PromptShield uses a **hybrid stacked ensemble architecture** that combines classical machine learning models with a Transformer-based deep learning model. The system consists of two main pipelines:

1. **Offline Training Pipeline** — trains all models and saves artifacts
2. **Online Inference Pipeline** — loads artifacts and classifies new prompts

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐        │
│  │  Dataset  │───►│ Preprocessing │───►│  Train/Test Split  │        │
│  │  (.xlsx)  │    │  + Cleaning   │    │   (80/20 strat.)   │        │
│  └──────────┘    └──────────────┘    └────────┬───────────┘        │
│                                                │                    │
│                                    ┌───────────▼───────────┐       │
│                                    │  Data Augmentation    │       │
│                                    │  (Training Only)      │       │
│                                    │  • Synonym Replace    │       │
│                                    │  • Random Deletion    │       │
│                                    │  • Random Swap        │       │
│                                    └───────────┬───────────┘       │
│                                                │                    │
│                           ┌────────────────────┼──────────┐        │
│                           │                    │          │        │
│                    ┌──────▼──────┐    ┌───────▼──┐  ┌───▼──────┐  │
│                    │  TF-IDF     │    │ RoBERTa  │  │ TF-IDF   │  │
│                    │  Vectorizer │    │ Tokenizer│  │ Vectorizer│  │
│                    └──┬──────┬──┘    └────┬─────┘  └──┬────┬──┘  │
│                       │      │            │           │    │      │
│                 ┌─────▼──┐ ┌─▼──────┐ ┌──▼────────┐  │    │      │
│                 │Random  │ │XGBoost │ │ RoBERTa   │  │    │      │
│                 │Forest  │ │        │ │ Fine-tune  │  │    │      │
│                 └────┬───┘ └───┬────┘ └─────┬─────┘  │    │      │
│                      │         │            │         │    │      │
│                      └────┬────┘────────────┘         │    │      │
│                           │                           │    │      │
│              ┌────────────▼─────────────┐             │    │      │
│              │  5-Fold Stratified OOF   │             │    │      │
│              │  Probability Collection  │             │    │      │
│              └────────────┬─────────────┘             │    │      │
│                           │                           │    │      │
│              ┌────────────▼─────────────┐             │    │      │
│              │  Logistic Regression     │             │    │      │
│              │  Meta-Classifier         │             │    │      │
│              └────────────┬─────────────┘             │    │      │
│                           │                           │    │      │
│              ┌────────────▼─────────────┐             │    │      │
│              │  Save Model Artifacts    │             │    │      │
│              │  (.pkl, model directory) │             │    │      │
│              └──────────────────────────┘             │    │      │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  User Prompt  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│  ┌──────▼──────────────┐                                            │
│  │  Harmful Keyword    │───── Match? ──► Return: Harmful (99%)      │
│  │  Check              │                                            │
│  └──────┬──────────────┘                                            │
│         │ No match                                                  │
│  ┌──────▼──────────────┐                                            │
│  │  Spell Correction   │                                            │
│  │  + Text Cleaning    │                                            │
│  └──────┬──────────────┘                                            │
│         │                                                           │
│         ├───── TF-IDF ──► RF Probs ──┐                              │
│         ├───── TF-IDF ──► XGB Probs ─┤                              │
│         └─── Tokenize ──► ROB Probs ─┘                              │
│                                      │                              │
│                          ┌───────────▼───────────┐                  │
│                          │  Concatenate (15-dim)  │                  │
│                          └───────────┬───────────┘                  │
│                                      │                              │
│                          ┌───────────▼───────────┐                  │
│                          │  Meta-Classifier       │                  │
│                          │  (Logistic Regression)  │                  │
│                          └───────────┬───────────┘                  │
│                                      │                              │
│                          ┌───────────▼───────────┐                  │
│                          │  Confidence Check     │                  │
│                          │  (threshold = 0.60)   │                  │
│                          └───────────┬───────────┘                  │
│                                      │                              │
│                          ┌───────────▼───────────┐                  │
│                          │  Final Prediction     │                  │
│                          │  (Label + Confidence) │                  │
│                          └───────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Preprocessing Module

- Converts text to lowercase
- Removes URLs using regex
- Strips special characters (keeps alphanumeric and spaces)
- Normalizes whitespace

### 2. Feature Extraction Module

- **TF-IDF Vectorizer**: Converts cleaned text to sparse numerical vectors
  - Max features: 3,000
  - N-gram range: (1, 2) — unigrams and bigrams
  - Sublinear TF scaling enabled
  - Minimum document frequency: 3

### 3. Base Model Layer

| Model | Input | Output |
|-------|-------|--------|
| Random Forest | TF-IDF vectors | 5-class probability distribution |
| XGBoost | TF-IDF vectors | 5-class probability distribution |
| RoBERTa | Tokenized text (max 128 tokens) | 5-class probability distribution (via softmax) |

### 4. Stacking Layer

- 5-fold stratified cross-validation generates out-of-fold predictions
- Probabilities from all 3 base models are concatenated into a 15-dimensional vector
- Logistic Regression meta-classifier is trained on these OOF features
- This prevents data leakage in the stacking process

### 5. Inference Safety Layer

- **Keyword matching**: First-pass check against a list of harmful keywords
- **Spell correction**: PySpellChecker corrects misspelled words (with skip list for tech terms)
- **Confidence thresholding**: Predictions below 60% confidence are flagged as uncertain

---

## Data Flow Summary

```
Input Prompt
    │
    ▼
[Keyword Check] ──(match)──► Harmful (100% confidence)
    │ (no match)
    ▼
[Spell Correct + Clean]
    │
    ├──► [TF-IDF] ──► [RF]      ──► 5 probabilities
    ├──► [TF-IDF] ──► [XGBoost] ──► 5 probabilities
    └──► [Tokenize] ► [RoBERTa] ──► 5 probabilities
                                        │
                                        ▼
                              [Concatenate: 15-dim]
                                        │
                                        ▼
                              [Meta-Classifier (LR)]
                                        │
                                        ▼
                              [Confidence Check]
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                         ≥ 0.60                < 0.60
                              │                    │
                         Final Label         UNCERTAIN
                         + Confidence       + Top-2 classes
                                           + Manual review flag
```
