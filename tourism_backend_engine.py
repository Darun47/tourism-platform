import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class TouristProfile:
    age: int
    interests: List[str]
    accessibility_needs: bool
    preferred_duration: int
    budget_preference: str
    climate_preference: Optional[str] = None


class TourismBackendEngine:

    def __init__(self, dataset_path):

        print("Loading dataset...")

        df = pd.read_csv(dataset_path)

        # normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # create compatibility column
        if "site_name" in df.columns:
            df["current_site"] = df["site_name"]

        self.df = df

        print("Dataset loaded:", len(self.df))


    # ============================================
    # ITINERARY GENERATION
    # ============================================

    def generate_itinerary(self, tourist_profile: TouristProfile):

        df = self.df.copy()

        if "budget_level" in df.columns:
            df = df[df["budget_level"] == tourist_profile.budget_preference]

        if "tourist_rating" in df.columns:
            df["score"] = df["tourist_rating"].fillna(3)
        else:
            df["score"] = 3

        df = df.sort_values("score", ascending=False)

        selected = df.head(tourist_profile.preferred_duration)

        days = []

        for i, (_, row) in enumerate(selected.iterrows(), start=1):

            days.append({
                "day": i,
                "city": row["city"],
                "site": row["current_site"],
                "cost": row.get("avg_cost_usd", 100)
            })

        return {
            "status": "success",
            "days": days
        }


    # ============================================
    # RECOMMENDATIONS
    # ============================================

    def get_recommendations(self, tourist_profile: TouristProfile):

        df = self.df.copy()

        if "budget_level" in df.columns:
            df = df[df["budget_level"] == tourist_profile.budget_preference]

        if "tourist_rating" in df.columns:
            df["score"] = df["tourist_rating"].fillna(3)
        else:
            df["score"] = 3

        df = df.sort_values("score", ascending=False)

        recs = []

        for _, row in df.head(5).iterrows():

            recs.append({
                "site": row["current_site"],
                "city": row["city"],
                "country": row["country"],
                "rating": row.get("tourist_rating", 0)
            })

        return recs


    # ============================================
    # ANALYTICS
    # ============================================

    def get_analytics(self):

        df = self.df

        return {
            "dataset_stats": {
                "total_records": len(df),
                "cities": df["city"].nunique(),
                "countries": df["country"].nunique()
            }
        }
