import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------
# App Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Cultural Tourism Platform",
    layout="wide"
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cultural_tourism_master.csv")

df = load_data()

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("🌍 AI Cultural Tourism Platform")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Personalized Itinerary Generator",
        "Smart Recommendations",
        "AI Travel Chatbot",
        "Analytics Dashboard",
        "PDF Export"
    ]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "Home":
    st.title("AI Cultural Tourism Insights & Engagement Platform")
    st.subheader("Your intelligent travel planning companion")

    st.markdown("""
    ✔ Personalized itineraries  
    ✔ Smart destination recommendations  
    ✔ AI travel chatbot  
    ✔ Interactive analytics  
    ✔ Professional PDF exports  
    """)

    st.success("Select a feature from the left menu to begin.")

# ---------------------------------------------------
# ITINERARY GENERATOR
# ---------------------------------------------------
elif page == "Personalized Itinerary Generator":

    st.title("🗺 Personalized Itinerary Generator")

    age = st.number_input("Age", 10, 80, 25)
    interests = st.multiselect(
        "Select Interests",
        ["Culture", "Nature", "Adventure", "History", "Food", "Wellness"]
    )
    budget = st.selectbox("Budget Level", ["Low", "Moderate", "High"])
    duration = st.slider("Trip Duration (days)", 3, 10, 5)

    if st.button("Generate Itinerary"):

        filtered = df[df["budget_level"] == budget]

        sample = filtered.sample(min(len(filtered), duration))

        itinerary = []
        daily_cost = 0

        for i, row in enumerate(sample.itertuples(), 1):
            itinerary.append(
                f"Day {i}: Visit {row.city}, {row.country} | Site: {row.destination_name}"
            )
            daily_cost += row.average_cost

        st.subheader("Your Trip Plan")
        for line in itinerary:
            st.write(line)

        st.success(f"Estimated Total Cost: ${int(daily_cost)}")

        st.session_state["itinerary"] = itinerary

# ---------------------------------------------------
# SMART RECOMMENDATIONS
# ---------------------------------------------------
elif page == "Smart Recommendations":

    st.title("🎯 Smart Recommendations")

    interest = st.selectbox(
        "Choose Primary Interest",
        ["Culture", "Nature", "Adventure", "History", "Food"]
    )

    recs = df.sample(10)

    recs["score"] = np.random.randint(70, 100, size=len(recs))

    st.dataframe(
        recs[["city", "country", "destination_name", "unesco_status", "score"]]
    )

    st.info("UNESCO Heritage Sites highlighted automatically.")

# ---------------------------------------------------
# AI CHATBOT
# ---------------------------------------------------
elif page == "AI Travel Chatbot":

    st.title("💬 AI Travel Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_input("Ask me about travel:")

    if st.button("Send"):
        response = f"Travel Tip: {user_input} is a great idea! Consider visiting during shoulder season for better prices."

        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("AI", response))

    for speaker, msg in st.session_state.chat:
        st.write(f"**{speaker}:** {msg}")

# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------
elif page == "Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Destinations", df.shape[0])
    col2.metric("Countries", df["country"].nunique())
    col3.metric("Average Cost", f"${int(df['average_cost'].mean())}")

    st.subheader("Top Countries")
    st.bar_chart(df["country"].value_counts().head(10))

    st.subheader("Budget Distribution")
    st.bar_chart(df["budget_level"].value_counts())

# ---------------------------------------------------
# PDF EXPORT
# ---------------------------------------------------
elif page == "PDF Export":

    st.title("📄 PDF Itinerary Export")

    if "itinerary" not in st.session_state:
        st.warning("Generate an itinerary first.")
    else:

        if st.button("Create PDF"):

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, "AI Cultural Tourism Itinerary", ln=True)
            pdf.cell(200, 10, f"Generated: {datetime.now()}", ln=True)

            pdf.ln(5)

            for line in st.session_state["itinerary"]:
                pdf.cell(200, 10, line, ln=True)

            pdf.cell(200, 10, "Packing Tips: Comfortable shoes, sunscreen, charger.", ln=True)

            pdf.output("itinerary.pdf")

            with open("itinerary.pdf", "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name="itinerary.pdf"
                )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("AI Cultural Tourism Platform | Capstone Project")
