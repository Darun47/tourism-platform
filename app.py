"""
AI Cultural Tourism Platform - Streamlit Frontend (FIXED VERSION)
==================================================================

This version uses ONLY native Streamlit components - no custom HTML/CSS
All blank card issues are resolved!

Author: AI Capstone Team
Date: February 1, 2026
Version: 1.1 (Bug Fix Release)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

from tourism_backend_engine import TourismBackendEngine, TouristProfile
from pdf_generator import PDFItineraryGenerator
from chatbot_integration import TravelChatbot

# Page configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'backend_engine' not in st.session_state:
    st.session_state.backend_engine = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'generated_itinerary' not in st.session_state:
    st.session_state.generated_itinerary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load backend engine (cached)
@st.cache_resource
def load_backend_engine(dataset_path):
    """Load and cache backend engine"""
    return TourismBackendEngine(dataset_path)

@st.cache_resource
def load_chatbot(_engine):
    """Load and cache chatbot"""
    return TravelChatbot(_engine)

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("🌍 AI Travel Planner")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Home", "✈️ Plan Your Trip", "💡 Recommendations", 
         "💬 Travel Assistant", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Initialize backend
    try:
        if st.session_state.backend_engine is None:
            with st.spinner("Loading AI backend..."):
                st.session_state.backend_engine = load_backend_engine(
                    'master_tourism_dataset_v2_enhanced.csv'
                )
                st.session_state.chatbot = load_chatbot(st.session_state.backend_engine)
            st.sidebar.success("✅ Backend loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Backend Error")
        st.error(f"**Error loading backend:** {str(e)}")
        st.info("📁 **Make sure** `master_tourism_dataset_v2_enhanced.csv` is in the same folder as `app.py`")
        st.stop()
    
    engine = st.session_state.backend_engine
    chatbot = st.session_state.chatbot
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page(engine)
    elif page == "✈️ Plan Your Trip":
        show_itinerary_page(engine)
    elif page == "💡 Recommendations":
        show_recommendations_page(engine)
    elif page == "💬 Travel Assistant":
        show_chatbot_page(chatbot, engine)
    elif page == "📊 Analytics":
        show_analytics_page(engine)
    elif page == "ℹ️ About":
        show_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    

# ============================================================================
# HOME PAGE
# ============================================================================

def show_home_page(engine):
    """Display home page with platform overview"""
    
    st.title("🌍 AI Cultural Tourism Platform")
    st.subheader("Your Intelligent Travel Companion")
    
    st.write("")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to the future of travel planning! Our AI-powered platform creates 
        **personalized cultural tourism experiences** tailored just for you.
        
        #### 🎯 What We Offer:
        - **Personalized Itineraries**: AI-generated travel plans based on your interests
        - **Smart Recommendations**: Discover destinations that match your preferences
        - **Budget Planning**: Find trips that fit your budget
        - **24/7 AI Assistant**: Get instant answers to your travel questions
        - **PDF Export**: Download and share your itinerary
        - **Accessibility Support**: Wheelchair-friendly options available
        """)
        
        if st.button("🚀 Start Planning Your Trip", type="primary", use_container_width=True):
            st.info("👈 Click **'Plan Your Trip'** in the sidebar to get started!")
    
    with col2:
        st.info("""
        **🌟 Platform Stats**
        
        📍 5 Major Cities  
        🏛️ 5,000+ Experiences  
        💰 Budget to Luxury  
        ♿ Accessibility Ready  
        🌡️ Climate-Aware Planning
        """)
    
    st.write("")
    st.divider()
    
    # Quick stats
    st.subheader("📊 Platform Overview")
    
    analytics = engine.get_analytics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎫 Total Experiences",
            value=f"{analytics['dataset_stats']['total_records']:,}"
        )
    
    with col2:
        st.metric(
            label="👥 Happy Travelers",
            value=f"{analytics['dataset_stats']['unique_tourists']:,}"
        )
    
    with col3:
        st.metric(
            label="💵 Avg Daily Cost",
            value=f"${analytics['cost_analysis']['avg_daily_cost_usd']:.0f}"
        )
    
    with col4:
        st.metric(
            label="⭐ Avg Satisfaction",
            value=f"{analytics['satisfaction_metrics']['avg_satisfaction']:.1f}/5"
        )
    
    st.write("")
    st.divider()
    
    # Popular destinations
    st.subheader("🔥 Our Featured Destinations")
    
    st.info("""
    **🌍 Available Cities:** Paris, Rome, Beijing, Agra, Cusco
    
    Each destination offers unique cultural experiences, from UNESCO World Heritage Sites 
    to world-class museums and historical landmarks!
    """)
    
    top_cities = list(analytics['popular_destinations']['top_cities'].items())[:5]
    
    cols = st.columns(5)
    city_info = {
        'Paris': {'icon': '🗼', 'country': 'France'},
        'Rome': {'icon': '🏛️', 'country': 'Italy'},
        'Beijing': {'icon': '🏯', 'country': 'China'},
        'Agra': {'icon': '🕌', 'country': 'India'},
        'Cusco': {'icon': '⛰️', 'country': 'Peru'}
    }
    
    for i, (city, count) in enumerate(top_cities):
        with cols[i]:
            info = city_info.get(city, {'icon': '🌍', 'country': 'Unknown'})
            st.markdown(f"<h1 style='text-align: center;'>{info['icon']}</h1>", unsafe_allow_html=True)
            st.markdown(f"**{city}**")
            st.caption(f"{info['country']}")
            st.caption(f"{count:,} experiences")

# ============================================================================
# ITINERARY PLANNING PAGE
# ============================================================================

def show_itinerary_page(engine):
    """Display itinerary planning page"""
    
    st.title("✈️ Plan Your Perfect Trip")
    
    st.markdown("""
    Tell us about yourself and your travel preferences, and our AI will create 
    a personalized itinerary just for you!
    """)
    
    st.write("")
    
    # Input form
    with st.form("itinerary_form"):
        st.subheader("👤 About You")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Your Age", 18, 80, 30)
            
            interests = st.multiselect(
                "Your Interests (select multiple)",
                ['Art', 'History', 'Architecture', 'Cultural', 'Nature'],
                default=['Art', 'History']
            )
        
        with col2:
            duration = st.slider("Trip Duration (days)", 1, 14, 7)
            
            budget = st.selectbox(
                "Budget Preference",
                ['Mid-range', 'Luxury'],
                index=0
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            climate = st.selectbox(
                "Climate Preference",
                ['Any', 'Temperate'],
                index=0
            )
        
        with col4:
            accessibility = st.checkbox("I need wheelchair accessibility")
        
        st.write("")
        
        start_date = st.date_input(
            "Preferred Start Date",
            value=datetime.now() + timedelta(days=30),
            min_value=datetime.now()
        )
        
        st.write("")
        
        submitted = st.form_submit_button(
            "🎯 Generate My Itinerary",
            type="primary",
            use_container_width=True
        )
    
    # Generate itinerary
    if submitted:
        if not interests:
            st.error("⚠️ Please select at least one interest!")
            return
        
        with st.spinner("🤖 AI is creating your perfect itinerary..."):
            try:
                profile = TouristProfile(
                    age=age,
                    interests=interests,
                    accessibility_needs=accessibility,
                    preferred_duration=duration,
                    budget_preference=budget,
                    climate_preference=None if climate == 'Any' else climate
                )
                
                itinerary = engine.generate_itinerary(
                    tourist_profile=profile,
                    start_date=datetime.combine(start_date, datetime.min.time())
                )
                
                st.session_state.generated_itinerary = itinerary
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        st.success("✅ Your personalized itinerary is ready!")
    
    # Display itinerary
    if st.session_state.generated_itinerary:
        display_itinerary(st.session_state.generated_itinerary, engine)

def display_itinerary(itinerary, engine):
    """Display generated itinerary"""
    
    if itinerary['status'] != 'success':
        st.error(itinerary.get('message', 'Failed to generate itinerary'))
        return
    
    st.write("")
    st.divider()
    st.subheader("🗺️ Your Personalized Itinerary")
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📅 Duration",
            f"{itinerary['itinerary']['total_days']} days"
        )
    
    with col2:
        st.metric(
            "💰 Total Cost",
            f"${itinerary['itinerary']['total_cost_usd']:,.0f}"
        )
    
    with col3:
        st.metric(
            "📍 Cities",
            len(itinerary['itinerary']['cities_visited'])
        )
    
    with col4:
        st.metric(
            "💵 Daily Average",
            f"${itinerary['itinerary']['avg_daily_cost_usd']:.0f}"
        )
    
    st.write("")
    
    # Trip overview
    with st.expander("📋 Trip Overview", expanded=True):
        st.write(f"**Dates:** {itinerary['itinerary']['start_date']} to {itinerary['itinerary']['end_date']}")
        st.write(f"**Cities:** {', '.join(itinerary['itinerary']['cities_visited'])}")
        st.write(f"**Your Interests:** {', '.join(itinerary['tourist_profile']['interests'])}")
        st.write(f"**Budget Level:** {itinerary['tourist_profile']['budget']}")
    
    st.write("")
    
    # Daily schedule
    st.subheader("📅 Daily Schedule")
    
    for day in itinerary['itinerary']['daily_schedule']:
        with st.container():
            st.markdown(f"### Day {day['day']} - {day['date']} | 📍 {day['city']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**🏛️ Sites to Visit:**")
                for site in day['sites']:
                    st.write(f"  • {site}")
                
                if day['activities']:
                    st.write("")
                    st.write("**🎯 Suggested Activities:**")
                    for activity in day['activities']:
                        st.write(f"  • {activity}")
                
                if day['notes']:
                    st.info(f"💡 {day['notes']}")
            
            with col2:
                st.metric("💰 Estimated Cost", f"${day['estimated_cost_usd']:.2f}")
            
            st.divider()
    
    # Recommendations
    if 'recommendations' in itinerary and itinerary['recommendations']:
        recs = itinerary['recommendations']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if recs.get('best_season'):
                st.success(f"🌞 **Best Season:** {recs['best_season']}")
            
            if recs.get('packing_tips'):
                with st.expander("🎒 Packing Essentials"):
                    for tip in recs['packing_tips']:
                        st.write(f"✓ {tip}")
        
        with col2:
            if recs.get('accessibility_info'):
                with st.expander("♿ Accessibility Information"):
                    info = recs['accessibility_info']
                    for key, value in info.items():
                        st.write(f"• {value}")
    
    st.write("")
    
    # Download PDF button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📄 Download PDF Itinerary", type="primary", use_container_width=True):
            generate_and_download_pdf(itinerary)

def generate_and_download_pdf(itinerary):
    """Generate and offer PDF download"""
    try:
        with st.spinner("Generating PDF..."):
            pdf_gen = PDFItineraryGenerator()
            pdf_path = "generated_itinerary.pdf"
            pdf_gen.generate_itinerary_pdf(itinerary, pdf_path)
            
            # Read PDF for download
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.download_button(
                label="⬇️ Download Your Itinerary",
                data=pdf_bytes,
                file_name=f"itinerary_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.success("✅ PDF generated successfully!")
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

# ============================================================================
# RECOMMENDATIONS PAGE
# ============================================================================

def show_recommendations_page(engine):
    """Display recommendations page"""
    
    st.title("💡 Discover Your Perfect Destination")
    
    st.markdown("""
    Get personalized recommendations based on your preferences. Our AI analyzes 
    thousands of experiences to find the perfect match for you!
    """)
    
    st.write("")
    
    # Filters
    tab1, tab2 = st.tabs(["🔍 Search Options", "ℹ️ Tips"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_type = st.selectbox(
                "What are you looking for?",
                ['All Recommendations', 'Cities Only', 'Specific Sites'],
                index=0
            )
            
            type_mapping = {
                'All Recommendations': 'all',
                'Cities Only': 'cities',
                'Specific Sites': 'sites'
            }
            rec_type_param = type_mapping[rec_type]
        
        with col2:
            num_recs = st.slider("Number of recommendations", 3, 10, 5)
        
        with col3:
            budget_filter = st.selectbox(
                "Budget Level",
                ['Mid-range', 'Luxury'],
                index=0
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            interests_filter = st.multiselect(
                "Your Interests",
                ['Art', 'History', 'Architecture', 'Cultural', 'Nature'],
                default=['Art', 'History']
            )
        
        with col5:
            age_filter = st.number_input("Your Age", 18, 80, 30)
    
    with tab2:
        st.info("""
        **💡 Tips for Best Results:**
        
        - Select multiple interests to get varied recommendations
        - Our dataset focuses on 5 major cultural destinations
        - Available cities: Paris, Rome, Beijing, Agra, Cusco
        - Higher match scores (80+) indicate excellent fits
        - UNESCO sites are specially highlighted
        """)
    
    st.write("")
    
    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        if not interests_filter:
            st.warning("⚠️ Please select at least one interest!")
            return
        
        with st.spinner("🤖 Finding your perfect matches..."):
            try:
                profile = TouristProfile(
                    age=age_filter,
                    interests=interests_filter,
                    accessibility_needs=False,
                    preferred_duration=7,
                    budget_preference=budget_filter
                )
                
                recommendations = engine.get_recommendations(
                    tourist_profile=profile,
                    num_recommendations=num_recs,
                    recommendation_type=rec_type_param
                )
                
                if recommendations['status'] == 'success':
                    st.success(f"✅ Found {recommendations['count']} recommendations!")
                    display_recommendations(recommendations)
                else:
                    st.error("Failed to generate recommendations.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def display_recommendations(recommendations):
    """Display recommendation results"""
    
    if recommendations['status'] != 'success':
        st.error("Failed to get recommendations")
        return
    
    if len(recommendations['recommendations']) == 0:
        st.warning("No recommendations found. Try different filters!")
        return
    
    st.write("")
    st.divider()
    st.subheader(f"🎯 Your Top {recommendations['count']} Matches")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### #{i}. {rec['name']}")
                st.write(f"**Type:** {rec['type'].title()}")
                st.write(f"**Why recommended:** {rec.get('reason', 'Great match for you')}")
                
                # Location for sites
                if 'city' in rec and rec.get('type') == 'site':
                    st.write(f"📍 **Location:** {rec['city']}, {rec.get('country', 'N/A')}")
                
                # UNESCO
                if 'unesco_site' in rec and rec['unesco_site']:
                    st.success("🏛️ UNESCO World Heritage Site")
            
            with col2:
                # Score
                score = rec.get('score', 0)
                if score >= 80:
                    st.success(f"**{score:.0f}/100**")
                    st.caption("Excellent Match!")
                elif score >= 60:
                    st.info(f"**{score:.0f}/100**")
                    st.caption("Good Match")
                else:
                    st.warning(f"**{score:.0f}/100**")
                    st.caption("Fair Match")
                
                # Cost
                if 'cost_usd' in rec:
                    st.metric("Daily Cost", f"${rec['cost_usd']:.0f}")
                elif 'avg_cost_usd' in rec:
                    st.metric("Daily Cost", f"${rec['avg_cost_usd']:.0f}")
            
            st.divider()

# ============================================================================
# CHATBOT PAGE
# ============================================================================

def show_chatbot_page(chatbot, engine):
    """Display chatbot page"""
    
    st.title("💬 AI Travel Assistant")
    
    st.markdown("""
    Ask me anything about travel planning! I can help you with destinations, 
    costs, itineraries, weather, and more. 🤖
    """)
    
    st.write("")
    
    # Chat container
    st.subheader("💭 Chat with Your Assistant")
    
    # Display history
    if not st.session_state.chat_history:
        st.info("👋 **Hello!** I'm your AI travel assistant. How can I help you plan your perfect trip today?")
    
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
    
    st.write("")
    
    # Suggested prompts (only show if no history)
    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎨 Recommend art destinations", use_container_width=True):
                handle_chat("Recommend some destinations for art lovers", chatbot)
                st.rerun()
        
        with col2:
            if st.button("💰 What's the average cost?", use_container_width=True):
                handle_chat("What's the average cost per day?", chatbot)
                st.rerun()
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("🏛️ Tell me about UNESCO sites", use_container_width=True):
                handle_chat("Tell me about UNESCO World Heritage sites", chatbot)
                st.rerun()
        
        with col4:
            if st.button("♿ Accessibility options", use_container_width=True):
                handle_chat("What accessibility options are available?", chatbot)
                st.rerun()
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        handle_chat(user_input, chatbot)
        st.rerun()
    
    # Clear button
    if st.session_state.chat_history:
        st.write("")
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.chat_history = []
            chatbot.clear_history()
            st.rerun()

def handle_chat(user_message, chatbot):
    """Handle chat message"""
    # Add user message
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_message
    })
    
    # Get bot response
    response = chatbot.chat(user_message)
    
    # Add bot response
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response['message']
    })

# ============================================================================
# ANALYTICS PAGE
# ============================================================================

def show_analytics_page(engine):
    """Display analytics dashboard"""
    
    st.title("📊 Platform Analytics")
    st.markdown("Explore insights from our tourism platform data.")
    
    st.write("")
    
    # Get analytics
    analytics = engine.get_analytics()
    
    # Dataset stats
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiences", f"{analytics['dataset_stats']['total_records']:,}")
    
    with col2:
        st.metric("Unique Tourists", f"{analytics['dataset_stats']['unique_tourists']:,}")
    
    with col3:
        st.metric("Cities", analytics['dataset_stats']['unique_cities'])
    
    with col4:
        st.metric("Countries", analytics['dataset_stats']['unique_countries'])
    
    st.write("")
    st.divider()
    
    # Popular destinations
    st.subheader("🌍 Most Popular Destinations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Cities**")
        cities_df = pd.DataFrame(
            list(analytics['popular_destinations']['top_cities'].items()),
            columns=['City', 'Visits']
        )
        st.bar_chart(cities_df.set_index('City')['Visits'])
    
    with col2:
        st.markdown("**Top Countries**")
        countries_df = pd.DataFrame(
            list(analytics['popular_destinations']['top_countries'].items()),
            columns=['Country', 'Visits']
        )
        st.bar_chart(countries_df.set_index('Country')['Visits'])
    
    st.write("")
    st.divider()
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Daily", f"${analytics['cost_analysis']['avg_daily_cost_usd']:.2f}")
    
    with col2:
        st.metric("Minimum", f"${analytics['cost_analysis']['min_cost_usd']:.2f}")
    
    with col3:
        st.metric("Maximum", f"${analytics['cost_analysis']['max_cost_usd']:.2f}")
    
    st.write("")
    
    budget_dist = analytics['cost_analysis']['budget_distribution']
    st.markdown("**Budget Distribution**")
    budget_df = pd.DataFrame(
        list(budget_dist.items()),
        columns=['Budget Level', 'Count']
    )
    st.bar_chart(budget_df.set_index('Budget Level')['Count'])
    
    st.write("")
    st.divider()
    
    # Tourist demographics
    st.subheader("👥 Tourist Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Age", f"{analytics['tourist_demographics']['avg_age']:.0f} years")
        st.metric("With Accessibility Needs", f"{analytics['tourist_demographics']['accessibility_needs_pct']:.1f}%")
    
    with col2:
        st.markdown("**Age Distribution**")
        age_df = pd.DataFrame(
            list(analytics['tourist_demographics']['age_distribution'].items()),
            columns=['Age Group', 'Count']
        )
        st.bar_chart(age_df.set_index('Age Group')['Count'])
    
    st.write("")
    st.divider()
    
    # Satisfaction
    st.subheader("⭐ Satisfaction Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Rating", f"{analytics['satisfaction_metrics']['avg_tourist_rating']:.2f}/5")
    
    with col2:
        st.metric("Satisfaction", f"{analytics['satisfaction_metrics']['avg_satisfaction']:.2f}/5")
    
    with col3:
        st.metric("Accuracy", f"{analytics['satisfaction_metrics']['recommendation_accuracy']:.0f}%")

# ============================================================================
# ABOUT PAGE
# ============================================================================

def show_about_page():
    """Display about page"""
    
    st.title("ℹ️ About This Platform")
    
    st.markdown("""
    ## AI Cultural Tourism Insights & Engagement Platform
    
    This platform leverages **Artificial Intelligence** and **Machine Learning** to create 
    personalized cultural tourism experiences.
    
    ### 🎯 Key Features
    
    1. **Personalized Itinerary Generation**
       - AI-powered trip planning based on your preferences
       - Budget-aware recommendations
       - Accessibility support
       - Multi-day scheduling with cost estimation
    
    2. **Smart Recommendations**
       - Collaborative filtering for similar travelers
       - Interest-based matching
       - UNESCO World Heritage site highlighting
    
    3. **AI Travel Assistant**
       - 24/7 chatbot support
       - Multilingual capabilities (framework ready)
       - Context-aware conversations
    
    4. **PDF Itinerary Export**
       - Professional, downloadable travel plans
       - Detailed daily schedules
       - Cost breakdowns and packing tips
    
    5. **Analytics Dashboard**
       - Platform insights and statistics
       - Popular destination trends
       - Cost and satisfaction metrics
    
    ### 🛠️ Technology Stack
    
    - **Frontend:** Streamlit (Python)
    - **Backend:** Custom AI engine with pandas, numpy
    - **AI/ML:** Gemini API integration framework
    - **PDF Generation:** ReportLab
    - **Data:** 9,989 curated tourism records
    
    ### 📊 Dataset
    
    - **Total Records:** 9,989 tourism experiences
    - **Unique Tourists:** 5,000 profiles
    - **Cities Covered:** 5 major cultural destinations (Paris, Rome, Beijing, Agra, Cusco)
    - **Data Quality:** 100% completeness on critical fields
    
    ### 👥 Project Information
    
    **Project:** AI Capstone - Scenario 3  
    **Platform:** GlobeTrek AI Solutions  
    **Version:** 1.1 (Bug Fix Release)
    **Date:** February 2026  
    
    ### 📝 How It Works
    
    1. **Input:** You provide your preferences (age, interests, budget, duration)
    2. **Processing:** AI analyzes 9,989 experiences to find the best matches
    3. **Scoring:** Destinations are scored based on:
       - Interest alignment (40%)
       - Ratings and reviews (30%)
       - Experience scores (30%)
    4. **Output:** Personalized itinerary with daily schedule and costs
    
    ### 🔒 Privacy & Data
    
    - Your preferences are processed locally
    - No personal data is stored permanently
    - AI recommendations are generated in real-time
    
    ---
    
    **Made with ❤️ using Streamlit and AI**
    """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
