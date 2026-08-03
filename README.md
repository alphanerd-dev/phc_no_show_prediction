# PHC No-Show Prediction

Capstone project for **3MTT × Nextgen** (Data Science Track)

## Problem Context
Missed primary healthcare (PHC) appointments waste clinical time and resources. This project builds a machine learning model to predict patient no-shows so clinics can apply proactive interventions (e.g., targeted reminders).

## MVP Coverage
This repository implements all required MVP items:

- **Data preparation**: date parsing, feature engineering (`wait_days`, `appointment_weekday`), numeric cleaning
- **Predictive model**: `LogisticRegression` in a preprocessing + modeling pipeline (scaled numeric features + one-hot categorical features)
- **Key no-show drivers**: top model coefficients, restricted to features with enough sample size per category to be statistically meaningful
- **Model evaluation**: accuracy, precision, recall, F1, ROC-AUC, and full classification report

## Repository Structure

- `data/KaggleV2-May-2016.csv` — Kaggle PHC appointment dataset (CSV)
- `src/train_model.py` — end-to-end training/evaluation script
- `artifacts/no_show_model.joblib` — trained model pipeline
- `artifacts/evaluation.json` — model evaluation metrics
- `artifacts/key_drivers.csv` — top model feature importances
- `requirements.txt` — Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python src/train_model.py
```

Optional arguments:

```bash
python src/train_model.py \
  --input-csv data/KaggleV2-May-2016.csv \
  --model-out artifacts/no_show_model.joblib \
  --evaluation-out artifacts/evaluation.json \
  --key-drivers-out artifacts/key_drivers.csv
```

## Current Evaluation Snapshot
From `artifacts/evaluation.json` (Kaggle dataset run, 110,527 appointments):

- Accuracy: **0.6666**
- Precision (No-show class): **0.318**
- Recall (No-show class): **0.5692**
- F1-score (No-show class): **0.4081**
- ROC-AUC: **0.6649**

Recall was prioritized over raw accuracy in model selection, since missing an actual no-show wastes a clinic slot, while a false alarm only costs an extra reminder call — a much cheaper mistake.

## Key Drivers (Top Features)
From `artifacts/key_drivers.csv`:

1. `wait_days` — days between booking and the appointment (strongest predictor by a clear margin)
2. `SMS_received` — whether the patient got a reminder
3. `Age`
4. Appointment weekday (Tuesday–Friday) — smaller, secondary effect

## Submission Checklist Mapping
- ✅ **Notebook/repo**: This repository contains the full implementation and outputs.
- ✅ **Trained model + evaluation**: Included in `artifacts/`.
- ✅ **README**: This document.

## Limitations
- Trained on a Brazilian appointments dataset used as a proxy — no public Nigerian PHC no-show dataset currently exists. A real deployment would need retraining on local PHC records.
- Doesn't capture transport distance or cost, which Nigerian studies point to as a major driver of missed appointments that this dataset can't represent.
