import streamlit as st
import pandas as pd

from tourism_backend_engine import TourismBackendEngine, TouristProfile
from pdf_generator import PDFItineraryGenerator
from chatbot_integration import TravelChatbot

st.set_page_config(page_title="AI Cultural Tourism Platform", layout="wide")

# Load backend
@st.cache_resource
def load_engine():
    return TourismBackendEngine("master_tourism_dataset_v2_enhanced.csv")

engine = load_engine()
chatbot = TravelChatbot(engine)

st.sidebar.title("🌍 AI Travel Planner")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Plan Trip", "Recommendations", "AI Chatbot", "Analytics"]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("AI Cultural Tourism Platform")

    st.write(
        "AI powered platform for personalized cultural tourism planning."
    )

    analytics = engine.get_analytics()

    col1, col2, col3 = st.columns(3)

    col1.metric("Experiences", analytics["dataset_stats"]["total_records"])
    col2.metric("Cities", analytics["dataset_stats"]["unique_cities"])
    col3.metric("Countries", analytics["dataset_stats"]["unique_countries"])


# ---------------- PLAN TRIP ----------------

elif page == "Plan Trip":

    st.header("Personalized Travel Itinerary")

    age = st.slider("Age", 18, 70, 30)

    interests = st.multiselect(
        "Travel Interests",
        ["Art", "History", "Nature", "Culture", "Architecture"]
    )

    duration = st.slider("Trip Duration", 1, 10, 5)

    budget = st.selectbox(
        "Budget Level",
        ["Budget", "Mid-range", "Luxury"]
    )

    if st.button("Generate Itinerary"):

        profile = TouristProfile(
            age=age,
            interests=interests,
            accessibility_needs=False,
            preferred_duration=duration,
            budget_preference=budget
        )

        itinerary = engine.generate_itinerary(profile)

        st.write(itinerary)

        if st.button("Download PDF"):

            pdf = PDFItineraryGenerator()

            pdf_path = pdf.generate_itinerary_pdf(itinerary)

            with open(pdf_path, "rb") as f:

                st.download_button(
                    "Download Itinerary PDF",
                    f,
                    "itinerary.pdf"
                )

# ---------------- RECOMMENDATIONS ----------------

elif page == "Recommendations":

    st.header("Destination Recommendations")

    interests = st.multiselect(
        "Interests",
        ["Art", "History", "Nature", "Culture"]
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
        st.write(r)


# ---------------- CHATBOT ----------------

elif page == "AI Chatbot":

    st.header("Travel AI Assistant")

    question = st.text_input("Ask travel question")

    if question:

        response = chatbot.chat(question)

        st.write(response)


# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    st.header("Tourism Analytics")

    analytics = engine.get_analytics()

    st.write(analytics)
