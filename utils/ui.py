from pathlib import Path
import streamlit as st
from database.db_connection import fetch_data
from utils.location_service import get_user_location, get_city_name
from datetime import datetime

def load_ui(page_title):

    st.set_page_config(
        page_title=page_title,
        page_icon="🍽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    css_path = Path("assets/style.css")

    with open(css_path, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

    # ==========================================
    # SIDEBAR
    # ==========================================

    with st.sidebar:

        try:
            st.image(
                "assets/logo.png",
                width=95
            )
        except:
            pass

        st.markdown(
            """
            <h2 style='text-align:center;margin-bottom:0'>
            🍽 AI Emotion-Aware
            </h2>

            <p style='text-align:center;
            color:#CBD5E1;
            margin-top:0'>
            Food Recommendation
            </p>

            <p style='text-align:center;
            font-size:13px;
            color:#CBD5E1'>
            Powered by AI • ViT • Hybrid ML
            </p>

            <hr>
            """,
            unsafe_allow_html=True
        )

        if "user_name" in st.session_state:

            st.success(
                f"👤 {st.session_state.user_name}"
            )

            st.caption("🟢 Online")

        st.divider()

        st.page_link(
            "app_main.py",
            label="🏠 Home"
        )

        st.page_link(
            "pages/3_Emotion_Detection.py",
            label="🎭 Emotion Detection"
        )

        st.page_link(
            "pages/4_Food_Recommendation.py",
            label="🍔 Food Recommendation"
        )

        st.page_link(
            "pages/5_Restaurant_Recommendation.py",
            label="🍴 Restaurant Recommendation"
        )

        st.page_link(
            "pages/6_Feedback.py",
            label="⭐ Feedback"
        )

        # st.page_link(
        #     "pages/7_Profile.py",
        #     label="👤 Profile"
        # )

        # st.page_link(
        #     "pages/8_History.py",
        #     label="📜 History"
        # )

        # st.page_link(
        #     "pages/9_Dashboard.py",
        #     label="📊 Dashboard"
        # )

        st.divider()

        if st.button(
            "🚪 Logout"
        ):

            st.session_state.clear()

            st.switch_page(
                "pages/1_Login.py"
            )

        st.markdown(
            """
            <br>

            <div style='text-align:center;
                        font-size:12px;
                        color:#CBD5E1'>

            Version 1.0

            <br>

            Final Year Project

            <br>

            University of Peshawar

            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# PAGE HEADER
# =====================================================

def page_header(title, icon="🍽"):

    username = st.session_state.get(
        "user_name",
        "User"
    )

    greeting = "Good Morning"

    hour = datetime.now().hour

    if hour >= 12:
        greeting = "Good Afternoon"

    if hour >= 17:
        greeting = "Good Evening"

    # -------------------------------
    # FIX: Initialize city here
    # -------------------------------
    city = "Unknown"

    user_id = st.session_state.get("user_id")

    if user_id is not None:

        lat, lon = get_user_location(user_id)

        if lat is not None and lon is not None:

            city_name = get_city_name(lat, lon)

            if city_name:
                city = city_name

    emotion = st.session_state.get(
        "emotion",
        "Not Detected"
    )

    budget = "-"

    if user_id is not None:

        profile = fetch_data(
            f"""
            SELECT monthly_food_budget
            FROM User_Profile
            WHERE user_id={user_id}
            """
        )

        if not profile.empty:

            budget = int(
                profile.iloc[0][
                    "monthly_food_budget"
                ]
            )

    st.markdown(
        f"""
<div style="

background:linear-gradient(
135deg,
#2563EB,
#7C3AED);

padding:30px;

border-radius:20px;

color:white;

margin-bottom:25px;

box-shadow:0 12px 30px rgba(0,0,0,.15);

">

<h1 style="color:white">

{icon} {title}

</h1>

<h3 style="color:white">

{greeting},

{username} 👋

</h3>

<hr>

<div style="display:flex;
justify-content:space-between;
flex-wrap:wrap">

<div>

📍 <b>{city}</b>

</div>

<div>

😊 <b>{emotion}</b>

</div>

<div>

💰 <b>Budget:
PKR {budget}</b>

</div>

</div>

</div>
""",
        unsafe_allow_html=True
    )
# =====================================================
# KPI CARD
# =====================================================

def kpi_card(

    icon,

    title,

    value,

    color="primary"

):

    color_class = {

        "primary":"",

        "success":"kpi-success",

        "warning":"kpi-warning",

        "danger":"kpi-danger"

    }.get(color,"")

    st.markdown(

        f"""

<div class="kpi-card">

<div class="kpi-icon">

{icon}

</div>

<div class="kpi-title">

{title}

</div>

<div class="kpi-value {color_class}">

{value}

</div>

</div>

""",

        unsafe_allow_html=True

    )


def page_intro(

    title,

    subtitle,

    emoji="🍽"

):

    st.markdown(

        f"""

<div class="login-card">

<h1 class="page-title">

{emoji} {title}

</h1>

<p class="page-subtitle">

{subtitle}

</p>

</div>

""",

        unsafe_allow_html=True

    )