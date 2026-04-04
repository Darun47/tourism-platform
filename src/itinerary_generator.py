from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd


def generate_itinerary(results: pd.DataFrame, start_date: str, days: int):
    if results is None or results.empty:
        return []

    try:
        date0 = datetime.fromisoformat(str(start_date)).date()
    except ValueError:
        date0 = datetime.today().date()

    choices = results["Site Name"].astype(str).tolist() if "Site Name" in results.columns else []
    if not choices:
        return []

    itinerary = []
    for i in range(days):
        itinerary.append({
            "date": str(date0 + timedelta(days=i)),
            "destination": choices[i % len(choices)],
        })

    return itinerary
