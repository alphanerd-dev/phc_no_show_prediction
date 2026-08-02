from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_COLUMNS = {
    "Gender", "ScheduledDay", "AppointmentDay", "Age", "Neighbourhood",
    "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
    "SMS_received", "No-show",
}

DEFAULT_INPUT_CSV = Path("data/KaggleV2-May-2016.csv")


def _prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = df.copy()
    data["ScheduledDay"] = pd.to_datetime(data["ScheduledDay"], errors="coerce", utc=True)
    data["AppointmentDay"] = pd.to_datetime(data["AppointmentDay"], errors="coerce", utc=True)

    data = data.dropna(subset=["ScheduledDay", "AppointmentDay"])
    data["wait_days"] = (data["AppointmentDay"] - data["ScheduledDay"]).dt.days.clip(lower=0)
    data["appointment_weekday"] = data["AppointmentDay"].dt.day_name()
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce").fillna(0).clip(lower=0)

    y = data["No-show"].astype(str).str.strip().str.lower().eq("yes").astype(int)

    # NOTE: Neighbourhood dropped from the driver-reporting model.
    # With 81 categories and some down to 1-2 appointments, one-hot coefficients
    # on it are unstable and misleading as "key drivers." Weekday and clinical/
    # demographic features are kept since they have enough data per category
    # to produce a stable, honest coefficient.
    feature_columns = [
        "Gender", "Age", "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "SMS_received", "wait_days", "appointment_weekday",
    ]
    x = data[feature_columns]
    return x, y


def train(input_csv: Path, model_out: Path, evaluation_out: Path, key_drivers_out: Path, test_size: float, random_state: int) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    x, y = _prepare_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_features = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "wait_days"]
    categorical_features = ["Gender", "appointment_weekday"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),  # FIX: puts numeric coefficients on the same scale as categorical dummies
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    model = LogisticRegression(random_state=random_state, class_weight="balanced", max_iter=1000, solver="liblinear")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_pred_proba = pipeline.predict_proba(x_test)[:, 1]

    evaluation = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "dataset_rows_used": int(len(x)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "no_show_rate": round(float(y.mean()), 4),
    }

    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    model_fitted = pipeline.named_steps["model"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = abs(model_fitted.coef_[0])

    key_drivers = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False).head(10).reset_index(drop=True)
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    evaluation_out.parent.mkdir(parents=True, exist_ok=True)
    key_drivers_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_out)
    evaluation_out.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    key_drivers.to_csv(key_drivers_out, index=False)
    print(json.dumps(evaluation, indent=2))
    print(key_drivers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PHC no-show prediction model")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/no_show_model.joblib"))
    parser.add_argument("--evaluation-out", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--key-drivers-out", type=Path, default=Path("artifacts/key_drivers.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        input_csv=args.input_csv, model_out=args.model_out, evaluation_out=args.evaluation_out,
        key_drivers_out=args.key_drivers_out, test_size=args.test_size, random_state=args.random_state,
    )
