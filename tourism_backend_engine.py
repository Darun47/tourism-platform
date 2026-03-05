import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class TouristProfile:

    age: int
    interests: List[str]
    accessibility_needs: bool
    preferred_duration: int
    budget_preference: str
    climate_preference: str = None


class TourismBackendEngine:

    def __init__(self, dataset_path):

        df = pd.read_csv(dataset_path)

        df.columns = df.columns.str.lower().str.replace(" ", "_")

        if "site_name" in df.columns:
            df["current_site"] = df["site_name"]

        self.df = df


    def generate_itinerary(self, profile: TouristProfile):

        df = self.df.copy()

        if "budget_level" in df.columns:
            df = df[df["budget_level"] == profile.budget_preference]

        df = df.sort_values("tourist_rating", ascending=False)

        selected = df.head(profile.preferred_duration)

        days = []

        for _, row in selected.iterrows():

            days.append({
                "city": row["city"],
                "site": row["current_site"],
                "cost": row["avg_cost_usd"]
            })

        return {"days": days}


    def get_recommendations(self, profile: TouristProfile):

        df = self.df.copy()

        df = df.sort_values("tourist_rating", ascending=False)

        top = df.head(5)

        recs = []

        for _, row in top.iterrows():

            recs.append({
                "name": row["current_site"],
                "city": row["city"],
                "country": row["country"],
                "rating": row["tourist_rating"]
            })

        return {"recommendations": recs}


    def get_analytics(self):

        df = self.df

        return {

            "dataset_stats": {
                "total_records": len(df),
                "unique_cities": df["city"].nunique(),
                "unique_countries": df["country"].nunique()
            }

        }
