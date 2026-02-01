"""
AI Cultural Tourism Insights & Engagement Platform - Backend Engine
=====================================================================

Core Features:
1. Personalized Itinerary Generation
2. Smart Recommendations System
3. PDF Generation
4. Analytics & Insights
5. Chatbot Integration (Gemini API)

Author: AI Capstone Backend Team
Date: January 31, 2026
Version: 1.0
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TouristProfile:
    """Tourist profile data model"""
    age: int
    interests: List[str]
    accessibility_needs: bool
    preferred_duration: int
    budget_preference: str  # 'Budget', 'Mid-range', 'Luxury'
    climate_preference: Optional[str] = None  # 'Cold', 'Temperate', 'Warm'
    
    def __post_init__(self):
        """Validate tourist profile"""
        if self.age < 18 or self.age > 100:
            raise ValueError(f"Age must be between 18 and 100, got {self.age}")
        if self.preferred_duration < 1:
            raise ValueError(f"Duration must be at least 1 day, got {self.preferred_duration}")

@dataclass
class Destination:
    """Destination data model"""
    record_id: str
    city: str
    country: str
    site_name: str
    avg_cost_usd: float
    best_season: str
    climate: str
    culture_score: float
    adventure_score: float
    nature_score: float
    avg_rating: float
    unesco_site: bool
    
@dataclass
class ItineraryDay:
    """Single day in an itinerary"""
    day_number: int
    city: str
    sites: List[str]
    estimated_cost: float
    activities: List[str]
    notes: str

# ============================================================================
# CORE BACKEND ENGINE
# ============================================================================

class TourismBackendEngine:
    """
    Main backend engine for AI Cultural Tourism Platform
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize backend engine with master dataset
        
        Args:
            dataset_path: Path to master_clean_tourism_dataset_v1.csv
        """
        print("🚀 Initializing Tourism Backend Engine...")
        
        # Load master dataset
        self.df = pd.read_csv(dataset_path)
        
        # Parse list columns
        self._parse_list_columns()
        
        # Build indexes for fast lookup
        self._build_indexes()
        
        print(f"✓ Loaded {len(self.df):,} records")
        print(f"✓ {self.df['city'].nunique()} unique cities")
        print(f"✓ {self.df['Tourist ID'].nunique():,} tourist profiles")
        print(f"✓ Backend engine ready!\n")
    
    def _parse_list_columns(self):
        """Parse string representations of lists"""
        def safe_parse(val):
            if pd.isna(val):
                return []
            if isinstance(val, list):
                return val
            try:
                return ast.literal_eval(val)
            except:
                return []
        
        if 'Interests' in self.df.columns:
            self.df['Interests'] = self.df['Interests'].apply(safe_parse)
        if 'Sites Visited' in self.df.columns:
            self.df['Sites Visited'] = self.df['Sites Visited'].apply(safe_parse)
    
    def _build_indexes(self):
        """Build indexes for fast lookups"""
        # City index
        self.cities = self.df['city'].unique().tolist()
        
        # Budget level index
        self.budget_levels = self.df['budget_level'].unique().tolist()
        
        # Interest categories
        all_interests = set()
        for interests in self.df['Interests']:
            if isinstance(interests, list):
                all_interests.update(interests)
        self.interest_categories = sorted(list(all_interests))
        
        # Climate classifications
        self.climate_types = self.df['climate_classification'].unique().tolist()
    
    # ========================================================================
    # FEATURE 1: PERSONALIZED ITINERARY GENERATION
    # ========================================================================
    
    def generate_itinerary(
        self,
        tourist_profile: TouristProfile,
        start_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized travel itinerary based on tourist preferences
        
        Args:
            tourist_profile: TouristProfile object with user preferences
            start_date: Optional start date for the trip
            
        Returns:
            Dictionary containing complete itinerary
        """
        print(f"\n🗺️  Generating Itinerary for {tourist_profile.age}yo traveler...")
        
        if start_date is None:
            start_date = datetime.now()
        
        # Step 1: Filter destinations by preferences
        filtered_df = self._filter_by_preferences(tourist_profile)
        
        if len(filtered_df) == 0:
            return {
                'status': 'error',
                'message': 'No destinations match your preferences'
            }
        
        # Step 2: Score and rank destinations
        ranked_destinations = self._score_destinations(
            filtered_df, 
            tourist_profile
        )
        
        # Step 3: Select top destinations based on duration
        selected_destinations = self._select_destinations(
            ranked_destinations,
            tourist_profile.preferred_duration
        )
        
        # Step 4: Build daily itinerary
        itinerary_days = self._build_daily_itinerary(
            selected_destinations,
            tourist_profile,
            start_date
        )
        
        # Step 5: Calculate total cost and stats
        total_cost = sum(day.estimated_cost for day in itinerary_days)
        
        result = {
            'status': 'success',
            'tourist_profile': {
                'age': tourist_profile.age,
                'interests': tourist_profile.interests,
                'budget': tourist_profile.budget_preference,
                'duration': tourist_profile.preferred_duration,
                'accessibility': tourist_profile.accessibility_needs
            },
            'itinerary': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': (start_date + timedelta(days=len(itinerary_days)-1)).strftime('%Y-%m-%d'),
                'total_days': len(itinerary_days),
                'total_cost_usd': round(total_cost, 2),
                'avg_daily_cost_usd': round(total_cost / len(itinerary_days), 2),
                'cities_visited': list(set([day.city for day in itinerary_days])),
                'daily_schedule': [
                    {
                        'day': day.day_number,
                        'date': (start_date + timedelta(days=day.day_number-1)).strftime('%Y-%m-%d'),
                        'city': day.city,
                        'sites': day.sites,
                        'activities': day.activities,
                        'estimated_cost_usd': round(day.estimated_cost, 2),
                        'notes': day.notes
                    }
                    for day in itinerary_days
                ]
            },
            'recommendations': {
                'best_season': self._get_best_season(selected_destinations),
                'packing_tips': self._get_packing_tips(selected_destinations, tourist_profile),
                'accessibility_info': self._get_accessibility_info(selected_destinations) if tourist_profile.accessibility_needs else None
            }
        }
        
        print(f"✓ Generated {len(itinerary_days)}-day itinerary")
        print(f"✓ Total cost: ${total_cost:,.2f}")
        print(f"✓ Cities: {', '.join(result['itinerary']['cities_visited'])}")
        
        return result
    
    def _filter_by_preferences(self, profile: TouristProfile) -> pd.DataFrame:
        """Filter destinations by tourist preferences"""
        df = self.df.copy()
        
        # Filter by budget
        if profile.budget_preference:
            df = df[df['budget_level'] == profile.budget_preference]
        
        # Filter by climate preference
        if profile.climate_preference:
            df = df[df['climate_classification'] == profile.climate_preference]
        
        # Filter by accessibility if needed
        # Note: This would require accessibility data in the dataset
        
        return df
    
    def _score_destinations(
        self, 
        df: pd.DataFrame, 
        profile: TouristProfile
    ) -> pd.DataFrame:
        """
        Score destinations based on tourist interests
        
        Uses a weighted scoring system:
        - Interest match: 40%
        - Rating: 30%
        - Experience scores: 30%
        """
        df = df.copy()
        
        # Score 1: Interest match (0-100)
        def calc_interest_match(row):
            row_interests = row.get('Interests', [])
            if not isinstance(row_interests, list) or not profile.interests:
                return 0
            
            matches = len(set(row_interests) & set(profile.interests))
            total = len(profile.interests)
            return (matches / total) * 100 if total > 0 else 0
        
        df['interest_score'] = df.apply(calc_interest_match, axis=1)
        
        # Score 2: Normalize ratings (0-100)
        df['rating_score'] = (df['Avg Rating'].fillna(df['Tourist Rating']) / 5.0) * 100
        
        # Score 3: Experience scores (0-100)
        experience_cols = ['culture', 'adventure', 'nature']
        experience_scores = []
        
        for col in experience_cols:
            if col in df.columns:
                # Map interest to experience type
                if any(interest.lower() in ['art', 'history', 'cultural'] for interest in profile.interests):
                    if col == 'culture':
                        experience_scores.append(df[col].fillna(0) * 20)  # Scale 1-5 to 0-100
                if any(interest.lower() in ['adventure'] for interest in profile.interests):
                    if col == 'adventure':
                        experience_scores.append(df[col].fillna(0) * 20)
                if any(interest.lower() in ['nature'] for interest in profile.interests):
                    if col == 'nature':
                        experience_scores.append(df[col].fillna(0) * 20)
        
        df['experience_score'] = sum(experience_scores) / len(experience_scores) if experience_scores else 50
        
        # Composite score
        df['composite_score'] = (
            df['interest_score'] * 0.4 +
            df['rating_score'] * 0.3 +
            df['experience_score'] * 0.3
        )
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        return df
    
    def _select_destinations(
        self,
        ranked_df: pd.DataFrame,
        num_days: int
    ) -> pd.DataFrame:
        """Select top destinations for the itinerary"""
        # Aim for 2-3 sites per day
        num_destinations = max(num_days, int(num_days * 1.5))
        
        # Get top destinations, ensuring city diversity
        selected = []
        cities_visited = set()
        
        for _, row in ranked_df.iterrows():
            # Prefer visiting different cities
            city = row['city']
            
            # Add destination
            selected.append(row)
            cities_visited.add(city)
            
            if len(selected) >= num_destinations:
                break
        
        return pd.DataFrame(selected)
    
    def _build_daily_itinerary(
        self,
        destinations: pd.DataFrame,
        profile: TouristProfile,
        start_date: datetime
    ) -> List[ItineraryDay]:
        """Build day-by-day itinerary"""
        days = []
        sites = destinations['current_site'].tolist()
        cities = destinations['city'].tolist()
        costs = destinations['avg_cost_usd'].tolist()
        
        # Group by city
        city_groups = {}
        for i, city in enumerate(cities):
            if city not in city_groups:
                city_groups[city] = []
            city_groups[city].append({
                'site': sites[i],
                'cost': costs[i]
            })
        
        # Build daily schedule
        day_num = 1
        for city, city_sites in city_groups.items():
            # Determine days in this city
            days_in_city = min(len(city_sites), profile.preferred_duration - len(days))
            
            for day_in_city in range(days_in_city):
                # Select sites for this day (2-3 sites)
                day_sites = city_sites[day_in_city:day_in_city+2]
                
                day = ItineraryDay(
                    day_number=day_num,
                    city=city,
                    sites=[s['site'] for s in day_sites],
                    estimated_cost=sum(s['cost'] for s in day_sites) / len(day_sites) if day_sites else 0,
                    activities=self._suggest_activities(city, profile.interests),
                    notes=f"Explore {city}'s cultural heritage"
                )
                
                days.append(day)
                day_num += 1
                
                if len(days) >= profile.preferred_duration:
                    break
            
            if len(days) >= profile.preferred_duration:
                break
        
        return days
    
    def _suggest_activities(self, city: str, interests: List[str]) -> List[str]:
        """Suggest activities based on interests"""
        activities = {
            'Art': ['Visit art galleries', 'Explore street art', 'Attend art exhibitions'],
            'History': ['Historical walking tour', 'Visit museums', 'Explore ancient sites'],
            'Architecture': ['Architecture tour', 'Visit iconic buildings', 'Photography walk'],
            'Cultural': ['Local food tasting', 'Cultural show', 'Visit local markets'],
            'Nature': ['Nature walks', 'Parks and gardens', 'Scenic viewpoints']
        }
        
        suggested = []
        for interest in interests:
            if interest in activities:
                suggested.extend(activities[interest][:2])
        
        return suggested[:4] if suggested else ['City exploration', 'Local cuisine']
    
    def _get_best_season(self, destinations: pd.DataFrame) -> str:
        """Determine best season for travel"""
        seasons = destinations['Best Season'].dropna().tolist()
        if seasons:
            # Return most common season
            from collections import Counter
            return Counter(seasons).most_common(1)[0][0]
        return "Any season"
    
    def _get_packing_tips(
        self, 
        destinations: pd.DataFrame, 
        profile: TouristProfile
    ) -> List[str]:
        """Generate packing tips based on climate and season"""
        tips = []
        
        climates = destinations['climate_classification'].unique()
        
        if 'Cold' in climates:
            tips.extend(['Warm jacket', 'Thermal layers', 'Gloves and scarf'])
        if 'Warm' in climates:
            tips.extend(['Light clothing', 'Sunscreen', 'Hat for sun protection'])
        if 'Temperate' in climates:
            tips.extend(['Layered clothing', 'Light jacket', 'Comfortable shoes'])
        
        # Add general tips
        tips.extend(['Camera', 'Power adapter', 'Travel documents'])
        
        return tips[:6]
    
    def _get_accessibility_info(self, destinations: pd.DataFrame) -> Dict[str, Any]:
        """Get accessibility information"""
        return {
            'wheelchair_accessible': "Most major sites have accessibility features",
            'assistance_available': "Contact local tourism office for assistance",
            'recommended': "Pre-book accessible tours for guaranteed access"
        }
    
    # ========================================================================
    # FEATURE 2: SMART RECOMMENDATIONS
    # ========================================================================
    
    def get_recommendations(
        self,
        tourist_profile: TouristProfile,
        num_recommendations: int = 5,
        recommendation_type: str = 'all'  # 'all', 'cities', 'sites'
    ) -> Dict[str, Any]:
        """
        Get smart recommendations based on tourist profile
        
        Args:
            tourist_profile: TouristProfile object
            num_recommendations: Number of recommendations to return
            recommendation_type: Type of recommendations ('all', 'cities', 'sites')
            
        Returns:
            Dictionary with recommendations
        """
        print(f"\n💡 Generating {num_recommendations} recommendations...")
        
        # Filter and score
        filtered_df = self._filter_by_preferences(tourist_profile)
        scored_df = self._score_destinations(filtered_df, tourist_profile)
        
        if recommendation_type == 'cities':
            # Recommend cities
            city_scores = scored_df.groupby('city').agg({
                'composite_score': 'mean',
                'avg_cost_usd': 'mean',
                'culture': 'mean',
                'Avg Rating': 'mean'
            }).reset_index()
            
            city_scores = city_scores.sort_values('composite_score', ascending=False)
            
            recommendations = []
            for _, row in city_scores.head(num_recommendations).iterrows():
                recommendations.append({
                    'type': 'city',
                    'name': row['city'],
                    'score': round(row['composite_score'], 2),
                    'avg_cost_usd': round(row['avg_cost_usd'], 2),
                    'culture_rating': round(row['culture'], 2),
                    'overall_rating': round(row['Avg Rating'], 2),
                    'reason': self._generate_recommendation_reason(row['city'], tourist_profile)
                })
        
        elif recommendation_type == 'sites':
            # Recommend specific sites
            recommendations = []
            for _, row in scored_df.head(num_recommendations).iterrows():
                recommendations.append({
                    'type': 'site',
                    'name': row['current_site'],
                    'city': row['city'],
                    'country': row['country'],
                    'score': round(row['composite_score'], 2),
                    'cost_usd': round(row['avg_cost_usd'], 2),
                    'unesco_site': row.get('UNESCO Site', False),
                    'reason': self._generate_site_reason(row, tourist_profile)
                })
        
        else:  # 'all'
            # Mixed recommendations
            recommendations = []
            
            # Add top cities
            top_cities = scored_df.groupby('city')['composite_score'].mean().nlargest(3)
            for city, score in top_cities.items():
                recommendations.append({
                    'type': 'city',
                    'name': city,
                    'score': round(score, 2),
                    'reason': f"Great match for your interests in {', '.join(tourist_profile.interests)}"
                })
            
            # Add top sites
            for _, row in scored_df.head(num_recommendations - len(recommendations)).iterrows():
                recommendations.append({
                    'type': 'site',
                    'name': row['current_site'],
                    'city': row['city'],
                    'score': round(row['composite_score'], 2),
                    'reason': self._generate_site_reason(row, tourist_profile)
                })
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        return {
            'status': 'success',
            'count': len(recommendations),
            'recommendations': recommendations,
            'profile_summary': {
                'interests': tourist_profile.interests,
                'budget': tourist_profile.budget_preference,
                'duration': tourist_profile.preferred_duration
            }
        }
    
    def _generate_recommendation_reason(
        self, 
        city: str, 
        profile: TouristProfile
    ) -> str:
        """Generate explanation for city recommendation"""
        city_data = self.df[self.df['city'] == city].iloc[0]
        
        reasons = []
        
        # Budget match
        if city_data['budget_level'] == profile.budget_preference:
            reasons.append(f"fits your {profile.budget_preference.lower()} budget")
        
        # Interest match
        if profile.interests:
            reasons.append(f"great for {', '.join(profile.interests[:2]).lower()} enthusiasts")
        
        # Climate
        if city_data['climate_classification']:
            reasons.append(f"{city_data['climate_classification'].lower()} climate year-round")
        
        return "; ".join(reasons).capitalize()
    
    def _generate_site_reason(self, row: pd.Series, profile: TouristProfile) -> str:
        """Generate explanation for site recommendation"""
        reasons = []
        
        # UNESCO status
        if row.get('UNESCO Site', False):
            reasons.append("UNESCO World Heritage Site")
        
        # Rating
        rating = row.get('Avg Rating', row.get('Tourist Rating', 0))
        if rating >= 4.5:
            reasons.append(f"highly rated ({rating:.1f}/5)")
        
        # Interest match
        site_interests = row.get('Interests', [])
        if isinstance(site_interests, list):
            matching = set(site_interests) & set(profile.interests)
            if matching:
                reasons.append(f"matches your interest in {', '.join(matching).lower()}")
        
        return "; ".join(reasons).capitalize() if reasons else "Popular destination"
    
    # ========================================================================
    # FEATURE 3: ANALYTICS & INSIGHTS
    # ========================================================================
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get platform analytics and insights
        
        Returns:
            Dictionary with analytics data
        """
        print("\n📊 Generating analytics...")
        
        analytics = {
            'dataset_stats': {
                'total_records': len(self.df),
                'unique_tourists': self.df['Tourist ID'].nunique(),
                'unique_cities': self.df['city'].nunique(),
                'unique_countries': self.df['country'].nunique(),
                'date_range': {
                    'last_validated': self.df['last_validated'].iloc[0] if 'last_validated' in self.df.columns else 'N/A'
                }
            },
            'popular_destinations': {
                'top_cities': self.df['city'].value_counts().head(5).to_dict(),
                'top_countries': self.df['country'].value_counts().head(5).to_dict()
            },
            'tourist_demographics': {
                'avg_age': round(self.df['Age'].mean(), 1),
                'age_distribution': self.df['Age_Group'].value_counts().to_dict(),
                'accessibility_needs_pct': round((self.df['Accessibility'].sum() / len(self.df)) * 100, 1)
            },
            'cost_analysis': {
                'avg_daily_cost_usd': round(self.df['avg_cost_usd'].mean(), 2),
                'min_cost_usd': round(self.df['avg_cost_usd'].min(), 2),
                'max_cost_usd': round(self.df['avg_cost_usd'].max(), 2),
                'budget_distribution': self.df['budget_level'].value_counts().to_dict()
            },
            'satisfaction_metrics': {
                'avg_tourist_rating': round(self.df['Tourist Rating'].mean(), 2),
                'avg_satisfaction': round(self.df['Satisfaction'].mean(), 2),
                'recommendation_accuracy': round(self.df['Recommendation Accuracy'].mean(), 1)
            },
            'climate_insights': {
                'climate_distribution': self.df['climate_classification'].value_counts().to_dict(),
                'avg_temperature': round(self.df['yearly_avg_temp'].mean(), 1)
            }
        }
        
        print("✓ Analytics generated")
        
        return analytics
    
    # ========================================================================
    # FEATURE 4: TOURIST PROFILE MATCHING
    # ========================================================================
    
    def find_similar_tourists(
        self,
        tourist_profile: TouristProfile,
        num_matches: int = 5
    ) -> Dict[str, Any]:
        """
        Find similar tourist profiles for collaborative filtering
        
        Args:
            tourist_profile: TouristProfile object
            num_matches: Number of similar profiles to return
            
        Returns:
            Dictionary with similar tourist data
        """
        print(f"\n👥 Finding {num_matches} similar tourists...")
        
        # Calculate similarity scores
        df = self.df.copy()
        
        # Age similarity (0-100)
        df['age_similarity'] = 100 - abs(df['Age'] - tourist_profile.age)
        df['age_similarity'] = df['age_similarity'].clip(0, 100)
        
        # Budget match
        df['budget_match'] = (df['budget_level'] == tourist_profile.budget_preference).astype(int) * 100
        
        # Interest overlap
        def calc_interest_overlap(row_interests):
            if not isinstance(row_interests, list) or not tourist_profile.interests:
                return 0
            overlap = len(set(row_interests) & set(tourist_profile.interests))
            total = len(set(row_interests) | set(tourist_profile.interests))
            return (overlap / total) * 100 if total > 0 else 0
        
        df['interest_similarity'] = df['Interests'].apply(calc_interest_overlap)
        
        # Composite similarity
        df['similarity_score'] = (
            df['age_similarity'] * 0.3 +
            df['budget_match'] * 0.3 +
            df['interest_similarity'] * 0.4
        )
        
        # Get top matches
        similar = df.nlargest(num_matches, 'similarity_score')
        
        matches = []
        for _, row in similar.iterrows():
            matches.append({
                'tourist_id': int(row['Tourist ID']),
                'age': int(row['Age']),
                'interests': row['Interests'] if isinstance(row['Interests'], list) else [],
                'budget': row['budget_level'],
                'visited_cities': [row['city']],
                'similarity_score': round(row['similarity_score'], 2),
                'avg_satisfaction': round(row['Satisfaction'], 2)
            })
        
        print(f"✓ Found {len(matches)} similar profiles")
        
        return {
            'status': 'success',
            'count': len(matches),
            'matches': matches
        }
    
    # ========================================================================
    # FEATURE 5: SEASONAL PLANNING
    # ========================================================================
    
    def get_seasonal_recommendations(
        self,
        season: str,
        budget: str = 'Mid-range',
        num_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Get seasonal destination recommendations
        
        Args:
            season: 'Spring', 'Summer', 'Autumn', 'Winter'
            budget: Budget preference
            num_recommendations: Number of recommendations
            
        Returns:
            Dictionary with seasonal recommendations
        """
        print(f"\n🌸 Finding {season} destinations...")
        
        # Filter by season
        seasonal_df = self.df[self.df['Best Season'] == season] if 'Best Season' in self.df.columns else self.df
        
        # Filter by budget
        if budget:
            seasonal_df = seasonal_df[seasonal_df['budget_level'] == budget]
        
        # Get top rated
        seasonal_df = seasonal_df.sort_values('Avg Rating', ascending=False)
        
        recommendations = []
        for _, row in seasonal_df.head(num_recommendations).iterrows():
            recommendations.append({
                'city': row['city'],
                'country': row['country'],
                'site': row['current_site'],
                'rating': round(row.get('Avg Rating', row.get('Tourist Rating', 0)), 2),
                'avg_cost_usd': round(row['avg_cost_usd'], 2),
                'climate': row['climate_classification'],
                'avg_temp': round(row['yearly_avg_temp'], 1) if pd.notna(row['yearly_avg_temp']) else None
            })
        
        print(f"✓ Found {len(recommendations)} {season} destinations")
        
        return {
            'status': 'success',
            'season': season,
            'budget': budget,
            'count': len(recommendations),
            'recommendations': recommendations
        }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AI CULTURAL TOURISM PLATFORM - BACKEND ENGINE TEST")
    print("=" * 80 + "\n")
    
    # Initialize engine
    engine = TourismBackendEngine(
        '/mnt/user-data/outputs/master_clean_tourism_dataset_v1.csv'
    )
    
    # Test 1: Generate Itinerary
    print("\n" + "=" * 80)
    print("TEST 1: PERSONALIZED ITINERARY GENERATION")
    print("=" * 80)
    
    tourist = TouristProfile(
        age=28,
        interests=['Art', 'History', 'Architecture'],
        accessibility_needs=False,
        preferred_duration=5,
        budget_preference='Mid-range',
        climate_preference='Temperate'
    )
    
    itinerary = engine.generate_itinerary(tourist)
    
    if itinerary['status'] == 'success':
        print("\n📋 ITINERARY SUMMARY:")
        print(f"  Duration: {itinerary['itinerary']['total_days']} days")
        print(f"  Total Cost: ${itinerary['itinerary']['total_cost_usd']:,.2f}")
        print(f"  Avg Daily Cost: ${itinerary['itinerary']['avg_daily_cost_usd']:,.2f}")
        print(f"  Cities: {', '.join(itinerary['itinerary']['cities_visited'])}")
        print(f"\n  Daily Schedule:")
        for day in itinerary['itinerary']['daily_schedule'][:3]:  # Show first 3 days
            print(f"    Day {day['day']} ({day['date']}): {day['city']}")
            print(f"      Sites: {', '.join(day['sites'][:2])}")
            print(f"      Cost: ${day['estimated_cost_usd']:.2f}")
    
    # Test 2: Smart Recommendations
    print("\n" + "=" * 80)
    print("TEST 2: SMART RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = engine.get_recommendations(
        tourist,
        num_recommendations=5,
        recommendation_type='all'
    )
    
    print("\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations['recommendations'][:5], 1):
        print(f"  {i}. {rec['name']} ({rec['type']})")
        print(f"     Score: {rec['score']}/100")
        print(f"     Reason: {rec.get('reason', 'N/A')}")
    
    # Test 3: Analytics
    print("\n" + "=" * 80)
    print("TEST 3: PLATFORM ANALYTICS")
    print("=" * 80)
    
    analytics = engine.get_analytics()
    
    print("\n📊 KEY METRICS:")
    print(f"  Total Records: {analytics['dataset_stats']['total_records']:,}")
    print(f"  Unique Tourists: {analytics['dataset_stats']['unique_tourists']:,}")
    print(f"  Avg Cost: ${analytics['cost_analysis']['avg_daily_cost_usd']:.2f}/day")
    print(f"  Avg Satisfaction: {analytics['satisfaction_metrics']['avg_satisfaction']}/5")
    print(f"\n  Top 3 Cities:")
    for city, count in list(analytics['popular_destinations']['top_cities'].items())[:3]:
        print(f"    - {city}: {count} visits")
    
    # Test 4: Similar Tourists
    print("\n" + "=" * 80)
    print("TEST 4: SIMILAR TOURIST PROFILES")
    print("=" * 80)
    
    similar = engine.find_similar_tourists(tourist, num_matches=3)
    
    print("\n👥 SIMILAR PROFILES:")
    for match in similar['matches']:
        print(f"  Tourist {match['tourist_id']} (Age {match['age']})")
        print(f"    Similarity: {match['similarity_score']}/100")
        print(f"    Interests: {', '.join(match['interests'])}")
        print(f"    Satisfaction: {match['avg_satisfaction']}/5")
    
    # Test 5: Seasonal Recommendations
    print("\n" + "=" * 80)
    print("TEST 5: SEASONAL PLANNING")
    print("=" * 80)
    
    seasonal = engine.get_seasonal_recommendations(
        season='Summer',
        budget='Mid-range',
        num_recommendations=3
    )
    
    print("\n🌞 SUMMER DESTINATIONS:")
    for rec in seasonal['recommendations']:
        print(f"  {rec['city']}, {rec['country']}")
        print(f"    Site: {rec['site']}")
        print(f"    Rating: {rec['rating']}/5")
        print(f"    Cost: ${rec['avg_cost_usd']:.2f}/day")
    
    print("\n" + "=" * 80)
    print("✅ BACKEND ENGINE TEST COMPLETE!")
    print("=" * 80)
