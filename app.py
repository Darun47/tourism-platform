import streamlit as st
from tourism_backend_engine import TourismBackendEngine, TouristProfile


st.set_page_config(page_title="AI Travel Planner")

st.title("🌍 AI Cultural Tourism Platform")

# load backend
@st.cache_resource
def load_engine():
    return TourismBackendEngine("master_tourism_dataset_v2_enhanced.csv")

engine = load_engine()


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Plan Trip", "Recommendations", "Analytics"]
)


# ===================================
# HOME
# ===================================

if page == "Home":

    st.header("Welcome to the AI Travel Planner")

    st.write(
        "This platform generates personalized cultural tourism recommendations."
    )


# ===================================
# PLAN TRIP
# ===================================

elif page == "Plan Trip":

    st.header("Generate Itinerary")

    age = st.slider("Age", 18, 70, 30)

    interests = st.multiselect(
        "Interests",
        ["Art", "History", "Nature", "Culture"]
    )

    duration = st.slider("Trip Duration", 1, 10, 5)

    budget = st.selectbox(
        "Budget",
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

        result = engine.generate_itinerary(profile)

        st.write(result)


# ===================================
# RECOMMENDATIONS
# ===================================

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

    st.write(recs)


# ===================================
# ANALYTICS
# ===================================

elif page == "Analytics":

    st.header("Dataset Analytics")

    data = engine.get_analytics()

    st.write(data)
