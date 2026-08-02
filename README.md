# PHC No-Show Prediction (DS-05)

Capstone project for **3MTT × Nextgen** (Data Science Track) by Byteforce Solutions Tech Ltd.

## Problem Context
Missed primary healthcare (PHC) appointments waste clinical time and resources. This project builds a machine learning model to predict patient no-shows so clinics can apply proactive interventions (e.g., targeted reminders).

## MVP Coverage
This repository implements all required MVP items:

- **Data preparation**: date parsing, feature engineering (`wait_days`, `appointment_weekday`), numeric cleaning
- **Predictive model**: `RandomForestClassifier` in a preprocessing + modeling pipeline
- **Key no-show drivers**: top feature importances exported to CSV
- **Model evaluation**: accuracy, precision, recall, F1, ROC-AUC, and full classification report

## Repository Structure

- `data/phc_appointments.csv` — sample PHC appointment dataset (CSV)
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
  --input-csv data/phc_appointments.csv \
  --model-out artifacts/no_show_model.joblib \
  --evaluation-out artifacts/evaluation.json \
  --key-drivers-out artifacts/key_drivers.csv
```

## Current Evaluation Snapshot
From `artifacts/evaluation.json` (sample run):

- Accuracy: **0.58**
- Precision (No-show class): **0.4737**
- Recall (No-show class): **0.45**
- F1-score (No-show class): **0.4615**
- ROC-AUC: **0.6594**

## Key Drivers (Top Features)
From `artifacts/key_drivers.csv`:

1. `wait_days`
2. `Age`
3. `SMS_received`
4. `Scholarship`
5. `Hipertension`

## Submission Checklist Mapping
- ✅ **Notebook/repo**: This repository contains the full implementation and outputs.
- ✅ **Trained model + evaluation**: Included in `artifacts/`.
- ✅ **README**: This document.
- ⏳ **2–3 min demo video**: Record a short walkthrough showing data flow, model training, metrics, and key drivers.

## Notes
- The included CSV is a runnable sample dataset to demonstrate the workflow end-to-end.
- Replace `data/phc_appointments.csv` with actual PHC appointment data to produce final deployment-ready results.
