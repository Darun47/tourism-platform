from __future__ import annotations

import pandas as pd


def choose_best_region(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if results is None or results.empty:
        return pd.DataFrame(columns=["Site Name", "city", "country"]), "Unknown"

    if "country" in results.columns and not results["country"].dropna().empty:
        country = results["country"].mode().iloc[0]
    else:
        country = "Unknown"

    return results, str(country)
