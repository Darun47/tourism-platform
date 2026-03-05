import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class TouristProfile:
    age: int
    interests: List[str]
    accessibility_needs: bool
    preferred_duration: int
    budget_preference: str
    climate_preference: Optional[str] = None


@dataclass
class ItineraryDay:
    day_number: int
    city: str
    sites: List[str]
    estimated_cost: float
    activities: List[str]
    notes: str


# =========================================================
# BACKEND ENGINE
# =========================================================

class TourismBackendEngine:

    def __init__(self, dataset_path: str):

        print("Loading tourism dataset...")

        df = pd.read_csv(dataset_path)

        # Normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Compatibility column
        if "site_name" in df.columns:
            df["current_site"] = df["site_name"]

        self.df = df

        print("Dataset loaded:", len(self.df), "rows")


    # =========================================================
    # ITINERARY GENERATION
    # =========================================================

    def generate_itinerary(self,
                           tourist_profile: TouristProfile,
                           start_date: Optional[datetime] = None
                           ) -> Dict[str, Any]:

        if start_date is None:
            start_date = datetime.now()

        df = self.df.copy()

        # Budget filter
        if "budget_level" in df.columns:
            df = df[df["budget_level"] == tourist_profile.budget_preference]

        # Rating score
        if "tourist_rating" in df.columns:
            df["score"] = df["tourist_rating"].fillna(3)
        else:
            df["score"] = 3

        df = df.sort_values("score", ascending=False)

        selected = df.head(tourist_profile.preferred_duration)

        itinerary_days = []

        for i, (_, row) in enumerate(selected.iterrows(), start=1):

            day = ItineraryDay(
                day_number=i,
                city=row["city"],
                sites=[row["current_site"]],
                estimated_cost=row.get("avg_cost_usd", 100),
                activities=self._suggest_activities(
                    row["city"],
                    tourist_profile.interests
                ),
                notes=f"Explore {row['city']}"
            )

            itinerary_days.append(day)

        total_cost = sum(d.estimated_cost for d in itinerary_days)

        return {
            "status": "success",
            "itinerary": {
                "total_days": len(itinerary_days),
                "total_cost_usd": round(total_cost, 2),
                "avg_daily_cost_usd": round(total_cost / len(itinerary_days), 2),
                "cities_visited": list({d.city for d in itinerary_days}),
                "daily_schedule": [
                    {
                        "day": d.day_number,
                        "city": d.city,
                        "sites": d.sites,
                        "activities": d.activities,
                        "estimated_cost_usd": d.estimated_cost,
                        "notes": d.notes
                    }
                    for d in itinerary_days
                ]
            }
        }


    # =========================================================
    # RECOMMENDATIONS
    # =========================================================

    def get_recommendations(self,
                            tourist_profile: TouristProfile,
                            num_recommendations: int = 5
                            ) -> Dict[str, Any]:

        df = self.df.copy()

        if "budget_level" in df.columns:
            df = df[df["budget_level"] == tourist_profile.budget_preference]

        if "tourist_rating" in df.columns:
            df["score"] = df["tourist_rating"].fillna(3)
        else:
            df["score"] = 3

        df = df.sort_values("score", ascending=False)

        recs = []

        for _, row in df.head(num_recommendations).iterrows():

            recs.append({
                "name": row["current_site"],
                "city": row["city"],
                "country": row["country"],
                "score": float(row["score"]),
                "cost_usd": float(row.get("avg_cost_usd", 0))
            })

        return {
            "status": "success",
            "count": len(recs),
            "recommendations": recs
        }


    # =========================================================
    # ANALYTICS
    # =========================================================

    def get_analytics(self):

        df = self.df

        return {

            "dataset_stats": {
                "total_records": len(df),
                "unique_tourists": df["tourist_id"].nunique()
                if "tourist_id" in df.columns else 0,
                "unique_cities": df["city"].nunique(),
                "unique_countries": df["country"].nunique()
            },

            "popular_destinations": {
                "top_cities": df["city"].value_counts().head(5).to_dict(),
                "top_countries": df["country"].value_counts().head(5).to_dict()
            },

            "cost_analysis": {
                "avg_daily_cost_usd": float(df["avg_cost_usd"].mean())
                if "avg_cost_usd" in df.columns else 0,
                "min_cost_usd": float(df["avg_cost_usd"].min())
                if "avg_cost_usd" in df.columns else 0,
                "max_cost_usd": float(df["avg_cost_usd"].max())
                if "avg_cost_usd" in df.columns else 0
            }
        }


    # =========================================================
    # ACTIVITIES
    # =========================================================

    def _suggest_activities(self, city, interests):

        activities = {
            "Art": ["Visit art galleries", "Museum tour"],
            "History": ["Historical walking tour", "Ancient landmarks"],
            "Nature": ["Parks and gardens", "Nature walk"],
            "Cultural": ["Local food tasting", "Visit markets"]
        }

        suggestions = []

        for interest in interests:
            if interest in activities:
                suggestions.extend(activities[interest])

        if not suggestions:
            suggestions = ["City exploration"]

        return suggestions[:3]
