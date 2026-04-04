import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ModuleNotFoundError:
    genai = None
    GENAI_AVAILABLE = False
import pandas as pd
from src.data_processing import load_dataset, preprocess_dataset
from src.recommender_engine import AIDestinationRecommender
from src.itinerary_generator import generate_itinerary
from src.location_optimizer import choose_best_region
from src.pdf_generator import generate_pdf
from src.feedback_system import (
    save_feedback,
    load_feedback,
    average_rating,
    most_liked_destinations,
    interest_trends
)

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="GlobeTrek AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GlobeTrek AI Travel Planner")


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = load_dataset()
df = preprocess_dataset(df)

# ----------------------------------------------------
# Session state for Gemini chat (Tab 3)
# ----------------------------------------------------

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []
if "gemini_api_valid" not in st.session_state:
    st.session_state.gemini_api_valid = GENAI_AVAILABLE and False
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

# ----------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Trip Planner",
    "🔎 Explore Destinations",
    "💬 AI Travel Assistant",
    "📊 Feedback Analytics"
])

# ====================================================
# TAB 1 — TRIP PLANNER
# ====================================================

with tab1:

    st.header("Plan Your Trip")

    # --------------------------------
    # Trip Basics
    # --------------------------------

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date")

    with col2:
        days = st.slider(
            "Trip Duration (days)",
            1,
            5,
            3
        )

    # --------------------------------
    # Traveler Preferences
    # --------------------------------

    col3, col4 = st.columns(2)

    with col3:
        climate = st.selectbox(
            "Preferred Climate",
            ["Any", "Tropical", "Temperate", "Cold"]
        )

    with col4:
        budget = st.selectbox(
            "Budget Level",
            ["Low", "Mid-range", "Luxury"]
        )

    col5, col6 = st.columns(2)

    with col5:
        age_group = st.selectbox(
            "Traveler Age Group",
            ["Teen", "Adult", "Family", "Senior"]
        )

    with col6:
        accessibility = st.checkbox(
            "Wheelchair Accessible Locations"
        )

    # --------------------------------
    # Travel Interests
    # --------------------------------

    st.subheader("Travel Interests")

    interests = st.multiselect(
        "Select interests",
        [
            "culture",
            "adventure",
            "nature",
            "beaches",
            "nightlife",
            "cuisine",
            "wellness",
            "urban",
            "seclusion"
        ]
    )

    # --------------------------------
    # Trip Description
    # --------------------------------

    user_query = st.text_input(
        "Describe your trip",
        placeholder="Example: cultural trip in Greece with good food"
    )

    combined_query = f"""
    Trip request: {user_query}
    Climate: {climate}
    Budget: {budget}
    Age group: {age_group}
    Accessibility: {accessibility}
    Interests: {', '.join(interests)}
    """

    # --------------------------------
    # Generate Trip Button
    # --------------------------------

    if st.button("Generate Trip Plan"):

        recommender = AIDestinationRecommender(df)

        results = recommender.recommend(combined_query)

        results, country = choose_best_region(results)

        # Save for feedback system
        st.session_state["results"] = results
        st.session_state["country"] = country
        st.session_state["query"] = combined_query
        st.session_state["interests"] = interests

        st.success(f"Recommended Country: {country}")

        st.subheader("Suggested Destinations")

        st.dataframe(results[
            ["Site Name", "city", "country"]
        ])

        itinerary = generate_itinerary(
            results,
            str(start_date),
            days
        )

        # Save itinerary
        st.session_state["itinerary"] = itinerary

        st.subheader("Your Travel Itinerary")

        for day in itinerary:

            st.write(
                f"📅 {day['date']} → {day['destination']}"
            )

        # --------------------------------
        # Generate PDF
        # --------------------------------

        pdf_path = generate_pdf(
            itinerary,
            start_date,
            days
        )

        with open(pdf_path, "rb") as file:

            st.download_button(
                label="📄 Download Travel Plan (PDF)",
                data=file,
                file_name="GlobeTrek_Travel_Plan.pdf",
                mime="application/pdf"
            )

        # --------------------------------
        # Feedback System
        # --------------------------------

if "results" in st.session_state:

    st.subheader("Rate This Travel Plan")

    rating = st.slider(
        "How useful was this recommendation?",
        1,
        5,
        3,
        key="rating_slider"
    )

    if st.button("Submit Feedback"):

        destinations = st.session_state["results"]["Site Name"].tolist()

        save_feedback(
            st.session_state["query"],
            st.session_state["country"],
            destinations,
            rating,
            st.session_state["interests"]
        )

        st.success("Thank you! Your feedback has been recorded.")


# ====================================================
# TAB 2 — DESTINATION EXPLORER
# ====================================================

with tab2:

    st.header("Explore Destinations")

    search_query = st.text_input(
        "Search destinations",
        placeholder="Example: beaches in Spain"
    )

    if st.button("Find Destinations"):

        recommender = AIDestinationRecommender(df)

        results = recommender.recommend(search_query)

        results, country = choose_best_region(results)

        st.success(f"Top Region: {country}")

        st.dataframe(results[[
            "Site Name",
            "city",
            "country",
            "Type"
        ]])


# ====================================================
# TAB 3 — AI TRAVEL ASSISTANT
# ====================================================

with tab3:

    st.header("💬 AI Travel Assistant")

    if not GENAI_AVAILABLE:
        st.warning("`google-generativeai` is not installed. Add it to requirements and redeploy to enable the AI assistant.")

    # ── API Key Section ──────────────────────────────
    with st.expander(
        "🔑 Gemini API Key" + (" ✅ Connected" if st.session_state.gemini_api_valid else " ⚠️ Not connected"),
        expanded=not st.session_state.gemini_api_valid
    ):
        st.markdown(
            "Get your free key at "
            "[aistudio.google.com](https://aistudio.google.com/app/apikey)",
        )

        key_input = st.text_input(
            "Paste your Gemini API key",
            type="password",
            placeholder="AIza...",
            key="gemini_key_input"
        )

        col_a, col_b = st.columns([1, 2])

        with col_a:
            if st.button("✔ Validate Key", type="primary"):
                if not GENAI_AVAILABLE:
                    st.error("Gemini SDK is unavailable in this environment.")
                elif key_input:
                    try:
                        genai.configure(api_key=key_input)
                        test = genai.GenerativeModel("gemini-2.5-flash")
                        test.generate_content("hi")
                        st.session_state.gemini_api_valid = True
                        st.session_state.gemini_key = key_input
                        st.success("API key validated! You can now chat.")
                        st.rerun()
                    except Exception as e:
                        st.session_state.gemini_api_valid = False
                        st.error(f"Invalid key: {e}")
                else:
                    st.warning("Please enter an API key first.")

        with col_b:
            model_choice = st.selectbox(
                "Gemini Model",
                ["gemini-2.5-flash"],
                key="gemini_model"
            )

    # ── Clear chat button ────────────────────────────
    if st.session_state.gemini_messages:
        if st.button("🗑 Clear conversation"):
            st.session_state.gemini_messages = []
            st.rerun()

    # ── Chat history display ─────────────────────────
    for msg in st.session_state.gemini_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Chat input ───────────────────────────────────
    if not st.session_state.gemini_api_valid:
        st.info("⬆️ Enter and validate your Gemini API key above to start chatting.")
    else:
        TRAVEL_SYSTEM_PROMPT = (
            "You are an expert AI travel assistant for GlobeTrek, a premium tourism company.\n"
            "Your role:\n"
            "- Suggest personalized travel destinations\n"
            "- Provide day-wise itineraries when asked\n"
            "- Include practical tips on budget, weather, transport, and food\n"
            "- Be concise, friendly, and helpful\n"
            "- When the user asks for a multi-day trip, structure the response day by day.\n"
        )

        user_input = st.chat_input("Ask me anything about travel...")

        if user_input:
            # Add user message
            st.session_state.gemini_messages.append(
                {"role": "user", "content": user_input}
            )

            # Build conversation history for multi-turn
            history = []
            for m in st.session_state.gemini_messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})

            try:
                genai.configure(api_key=st.session_state.gemini_key)
                model = genai.GenerativeModel(
                    st.session_state.gemini_model,
                    system_instruction=TRAVEL_SYSTEM_PROMPT,
                )
                chat_session = model.start_chat(history=history)
                response = chat_session.send_message(user_input)
                reply = response.text.strip() if response.text else "No response. Please try again."
            except Exception as e:
                reply = f"⚠️ Error: {e}"

            st.session_state.gemini_messages.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()

# ====================================================
# TAB 4 — FEEDBACK ANALYTICS
# ====================================================

with tab4:

    st.header("User Feedback Analytics")

    df_feedback = load_feedback()

    if df_feedback is None or df_feedback.empty:

        st.info("No feedback data available yet.")

    else:

        # Average Rating
        avg_rating = average_rating(df_feedback)

        st.metric("Average User Rating", round(avg_rating, 2))

        # Most Liked Destinations
        st.subheader("Most Liked Destinations")

        liked_destinations = most_liked_destinations(df_feedback)

        if liked_destinations is not None:
            st.bar_chart(liked_destinations)

        # Interest Trends
        st.subheader("User Interest Trends")

        interest_data = interest_trends(df_feedback)

        if interest_data is not None:
            st.bar_chart(interest_data)

        # Debug (optional)
        st.dataframe(df_feedback)
