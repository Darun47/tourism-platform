from __future__ import annotations

from pathlib import Path
import pandas as pd


DATASET_FILE = "master_tourism_dataset_v2_enhanced.csv"


def load_dataset(dataset_path: str | None = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else Path(__file__).resolve().parent.parent / DATASET_FILE
    return pd.read_csv(path)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed.columns = [col.strip() for col in processed.columns]

    for col in ["Site Name", "city", "country", "Type", "Interests", "budget_level", "climate_classification"]:
        if col in processed.columns:
            processed[col] = processed[col].astype(str).fillna("")

    if "Tourist Rating" in processed.columns:
        processed["Tourist Rating"] = pd.to_numeric(processed["Tourist Rating"], errors="coerce").fillna(0.0)

    return processed
