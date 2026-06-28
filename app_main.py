# =====================================================
# APP.PY
# Personalized Emotion-Aware Food Recommendation System
# =====================================================

import streamlit as st


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Personalized Emotion-Aware Food Recommendation Applicaion",
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================


if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "emotion" not in st.session_state:
    st.session_state.emotion = None

if "emotion_id" not in st.session_state:
    st.session_state.emotion_id = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "selected_food" not in st.session_state:
    st.session_state.selected_food = None

if "selected_restaurant" not in st.session_state:
    st.session_state.selected_restaurant = None



# =====================================================
# MAIN PAGE
# =====================================================

st.title("🍽️ Personalized Emotion-Aware Food Recommendation System using ViT & Collaborative Filtering")

st.markdown("---")

st.write(
    """
    Welcome to the Personalized Emotion-Aware Food Recommendation System.

    This system recommends foods and restaurants based on:

    • Detected Emotion (ViT)

    • Personal Health Records (PHR)

    • Food Preferences

    • Budget Constraints

    • Collaborative Filtering

    • Feedback Learning

    • TOPSIS Restaurant Ranking
    """
)

st.markdown("---")


# =====================================================
# PROJECT STATUS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Foods", "269")

with col2:
    st.metric("Users", "500")

with col3:
    st.metric("Restaurants", "50")


st.markdown("---")

st.success(
    "Use the sidebar to navigate through the application."
)