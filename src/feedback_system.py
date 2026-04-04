from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


FEEDBACK_FILE = Path("feedback_log.csv")


def save_feedback(query, country, destinations, rating, interests):
    entry = {
        "query": query,
        "country": country,
        "destinations": json.dumps(destinations),
        "rating": int(rating),
        "interests": json.dumps(interests),
    }

    if FEEDBACK_FILE.exists():
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(FEEDBACK_FILE, index=False)


def load_feedback():
    if not FEEDBACK_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(FEEDBACK_FILE)


def average_rating(df_feedback: pd.DataFrame):
    if df_feedback.empty or "rating" not in df_feedback.columns:
        return 0.0
    return pd.to_numeric(df_feedback["rating"], errors="coerce").fillna(0).mean()


def most_liked_destinations(df_feedback: pd.DataFrame):
    if df_feedback.empty or "destinations" not in df_feedback.columns:
        return None

    all_destinations = []
    for raw in df_feedback["destinations"].dropna().astype(str):
        try:
            all_destinations.extend(json.loads(raw))
        except json.JSONDecodeError:
            continue

    if not all_destinations:
        return None

    return pd.Series(all_destinations).value_counts().head(10)


def interest_trends(df_feedback: pd.DataFrame):
    if df_feedback.empty or "interests" not in df_feedback.columns:
        return None

    all_interests = []
    for raw in df_feedback["interests"].dropna().astype(str):
        try:
            all_interests.extend(json.loads(raw))
        except json.JSONDecodeError:
            continue

    if not all_interests:
        return None

    return pd.Series(all_interests).value_counts().head(10)
