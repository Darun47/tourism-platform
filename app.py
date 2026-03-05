import streamlit as st
import pandas as pd
from datetime import datetime

from tourism_backend_engine import TourismBackendEngine, TouristProfile
from pdf_generator import PDFItineraryGenerator
from chatbot_integration import TravelChatbot

st.set_page_config(page_title="AI Cultural Tourism Platform", layout="wide")

# ---------------- LOAD ENGINE ----------------

@st.cache_resource
def load_engine():
    return TourismBackendEngine("master_tourism_dataset_v2_enhanced.csv")

engine = load_engine()
chatbot = TravelChatbot(engine)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌍 AI Travel Planner")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Plan Trip", "Recommendations", "Chatbot", "Analytics"]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("AI Cultural Tourism Insights Platform")

    st.write(
        """
        This platform generates personalized travel plans
        using AI-based recommendation systems.
        """
    )

    analytics = engine.get_analytics()

    col1, col2, col3 = st.columns(3)

    col1.metric("Experiences", analytics["dataset_stats"]["total_records"])
    col2.metric("Cities", analytics["dataset_stats"]["unique_cities"])
    col3.metric("Countries", analytics["dataset_stats"]["unique_countries"])


# ---------------- PLAN TRIP ----------------

elif page == "Plan Trip":

    st.title("✈️ Personalized Travel Planner")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Traveler Age", 18, 80, 30)

        interests = st.multiselect(
            "Travel Interests",
            ["Art", "History", "Architecture", "Culture", "Nature"]
        )

        duration = st.slider("Trip Duration (days)", 1, 14, 5)

        climate = st.selectbox(
            "Climate Preference",
            ["Any", "Cold", "Temperate", "Warm"]
        )

    with col2:

        budget = st.selectbox(
            "Budget Level",
            ["Budget", "Mid-range", "Luxury"]
        )

        accessibility = st.checkbox(
            "Wheelchair accessibility required"
        )

        start_date = st.date_input("Preferred Start Date", datetime.today())

    st.divider()

    if st.button("Generate My Itinerary"):

        profile = TouristProfile(
            age=age,
            interests=interests,
            accessibility_needs=accessibility,
            preferred_duration=duration,
            budget_preference=budget,
            climate_preference=climate
        )

        itinerary = engine.generate_itinerary(profile)

        st.success("Your itinerary is ready!")

        for i, day in enumerate(itinerary["days"], start=1):

            st.markdown(f"### Day {i} — {day['city']}")

            col1, col2 = st.columns(2)

            col1.write("🏛 Site:", day["site"])
            col2.write("💰 Estimated Cost:", f"${day['cost']}")

            st.divider()

        # PDF Export
        pdf = PDFItineraryGenerator()
        pdf_path = pdf.generate_itinerary_pdf(itinerary)

        with open(pdf_path, "rb") as f:

            st.download_button(
                "Download PDF Itinerary",
                f,
                "travel_plan.pdf"
            )


# ---------------- RECOMMENDATIONS ----------------

elif page == "Recommendations":

    st.title("💡 Smart Destination Recommendations")

    interests = st.multiselect(
        "Your Interests",
        ["Art", "History", "Architecture", "Culture", "Nature"]
    )

    profile = TouristProfile(
        age=30,
        interests=interests,
        accessibility_needs=False,
        preferred_duration=5,
        budget_preference="Mid-range"
    )

    recs = engine.get_recommendations(profile)

    for r in recs["recommendations"]:

        st.write(
            f"📍 {r['name']} — {r['city']}, {r['country']} | ⭐ {r['rating']}"
        )


# ---------------- CHATBOT ----------------

elif page == "Chatbot":

    st.title("💬 AI Travel Assistant")

    user_question = st.text_input("Ask a travel question")

    if user_question:

        response = chatbot.chat(user_question)

        st.write(response)


# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    st.title("📊 Tourism Data Analytics")

    analytics = engine.get_analytics()

    st.write(analytics)
