# =====================================================
# APP.PY
# Personalized Emotion-Aware Food Recommendation System
# =====================================================

import streamlit as st


# =====================================================
# PAGE CONFIG
# =====================================================

# st.set_page_config(
#     page_title="Personalized Emotion-Aware Food Recommendation Applicaion",
#     page_icon="🍽️", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
from utils.ui import load_ui

load_ui("Emotion Detection")



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



from PIL import Image

# =====================================================
# HOME PAGE
# =====================================================

logo = Image.open("assets/logo.png")

col1, col2, col3 = st.columns([1,3,1])

with col2:
    st.image(
        logo,
        use_container_width=True
    )

st.markdown(
"""
<h1 style="text-align:center; color:#FFFFFF;">

Personalized Emotion-Aware Food Recommendation System

</h1>

<h4 style="text-align:center;color:#FFF000;">

Powered by Vision Transformer (ViT) • Hybrid AI • Collaborative Filtering • TOPSIS

</h4>

<p style="text-align:center;
font-size:18px;
color:#475569;
max-width:900px;
margin:auto;">

An intelligent AI-based recommendation system that detects
human emotions using Vision Transformer (ViT) and recommends
personalized foods and nearby restaurants based on health profile,
preferences, budget, collaborative filtering, and real-time location.

</p>

""",
unsafe_allow_html=True
)

st.divider()

# =====================================================
# QUICK STATS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🍽 Foods", "269")

with col2:
    st.metric("👤 Users", "500+")

with col3:
    st.metric("🍴 Restaurants", "50+")

with col4:
    st.metric("🤖 AI Model", "ViT")

st.divider()

# =====================================================
# FEATURES
# =====================================================

st.subheader("✨ System Features")

c1, c2 = st.columns(2)

with c1:

    st.info("""
### 🎭 Emotion Detection

Detects user emotion using a Vision Transformer (ViT)
from an uploaded or captured facial image.
""")

    st.info("""
### 🍔 Food Recommendation

Hybrid recommendation using

• Emotion

• Health Profile

• Preferences

• Budget

• Collaborative Filtering
""")

with c2:

    st.info("""
### 🍴 Restaurant Recommendation

Nearby restaurants ranked using

• Distance

• Rating

• Price

• Delivery Time

• TOPSIS
""")

    st.info("""
### ⭐ Feedback Learning

User ratings improve future recommendations
through feedback learning.
""")

st.divider()

# =====================================================
# GET STARTED
# =====================================================

st.subheader("🚀 Ready to Get Started?")

col1, col2, col3 = st.columns([1,2,1])

with col2:

    if st.button(
        "🔐 Login to Continue",
        use_container_width=True
    ):
        st.switch_page(
            "pages/1_Login.py"
        )

st.divider()

# =====================================================
# FOOTER
# =====================================================

st.markdown(
"""
<div style="text-align:center;color:black;font-size:14px;">

Personalized Emotion-Aware Food Recommendation System

Final Year Project

Department of Computer Science

University of Peshawar

</div>
""",
unsafe_allow_html=True
)