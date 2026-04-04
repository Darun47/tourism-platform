from __future__ import annotations

import re
import pandas as pd


class AIDestinationRecommender:
    def __init__(self, dataset: pd.DataFrame):
        self.df = dataset.copy()

    def recommend(self, query: str, limit: int = 10) -> pd.DataFrame:
        if self.df.empty:
            return self.df

        query = (query or "").lower().strip()
        if not query:
            ranked = self.df.sort_values("Tourist Rating", ascending=False)
            return ranked.head(limit)

        tokens = [t for t in re.findall(r"[a-zA-Z]+", query) if len(t) > 2]
        scored = self.df.copy()
        scored["_score"] = 0.0

        searchable_cols = [c for c in ["Site Name", "city", "country", "Type", "Interests", "budget_level", "climate_classification"] if c in scored.columns]
        for token in tokens:
            token_score = 0
            for col in searchable_cols:
                token_score += scored[col].astype(str).str.lower().str.contains(token, regex=False).astype(int)
            scored["_score"] += token_score

        if "Tourist Rating" in scored.columns:
            scored["_score"] += pd.to_numeric(scored["Tourist Rating"], errors="coerce").fillna(0) / 5.0

        ranked = scored.sort_values(["_score", "Tourist Rating"], ascending=False)
        return ranked.drop(columns=["_score"]).head(limit)
