import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)



# =====================================================
# REGISTRATION WIZARD
# =====================================================

import streamlit as st
import bcrypt

from database.db_connection import (
    fetch_data,
    execute_query
)

# =====================================================
# PAGE TITLE
# =====================================================

# st.title("📝 User Registration")
from utils.ui import (
    load_ui,
    page_intro
)

load_ui("Registration Page")

page_intro(
    "Create Account",
    "Join the AI Emotion-Aware Food Recommendation System.",
    "📝"
)

# =====================================================
# INITIALIZE REGISTRATION SESSION
# =====================================================

if "reg_step" not in st.session_state:
    st.session_state.reg_step = 1

if "account" not in st.session_state:
    st.session_state.account = {}

if "phr" not in st.session_state:
    st.session_state.phr = {}

if "profile" not in st.session_state:
    st.session_state.profile = {}

# ===============================================
# REGISTRATION PROGRESS
# ===============================================

progress = {
    1: 25,
    2: 50,
    3: 75,
    4: 100
}

st.progress(progress[st.session_state.reg_step])

step_titles = {
    1: "👤 Account Information",
    2: "❤️ Health Profile",
    3: "🍽 Food Preferences",
    4: "✅ Review & Register"
}
st.markdown(
    f"""
    <div style="
        text-align:center;
        font-size:22px;
        font-weight:600;
        color:#2563EB;
        margin-bottom:20px;
    ">
        Step {st.session_state.reg_step} of 4
        <br><br>
        {step_titles[st.session_state.reg_step]}
    </div>
    """,
    unsafe_allow_html=True
)
# =====================================================
# SESSION INITIALIZATION
# =====================================================

if "reg_step" not in st.session_state:
    st.session_state.reg_step = 1

# =====================================================
# GET NEXT USER ID
# =====================================================

def get_next_user_id():

    result = fetch_data(
        """
        SELECT MAX(user_id) AS max_id
        FROM Users
        """
    )

    return int(result.iloc[0]["max_id"]) + 1


# =====================================================
# STEP 1
# ACCOUNT INFORMATION
# =====================================================
if st.session_state.reg_step == 1:
    left,right = st.columns(2)

    with left:

        full_name = st.text_input(
            "👤 Full Name"
        )

        age = st.number_input(
            "🎂 Age",
            10,
            100,
            20
        )

        gender = st.selectbox(
            "🚻 Gender",
            [
                "Male",
                "Female"
            ]
        )

        city = st.text_input(
            "📍 City"
        )

    with right:

        email = st.text_input(
            "📧 Email"
        )

        password = st.text_input(
            "🔒 Password",
            type="password"
        )

        confirm_password = st.text_input(
            "🔑 Confirm Password",
            type="password"
        )

        occupation = st.text_input(
            "💼 Occupation"
        )

        if st.button("Next ➜",
                     key="step1_next"
        ):

            if not full_name.strip():

                st.warning(
                    "⚠ Please enter Full Name."
                )

            elif not email.strip():

                st.warning(
                    "⚠ Please enter Email."
                )

            elif "@" not in email:

                st.warning(
                    "⚠ Invalid Email Address."
                )

            elif len(password) < 6:

                st.warning(
                    "⚠ Password must be at least 6 characters."
                )

            elif password != confirm_password:

                st.warning(
                    "⚠ Passwords do not match."
                )

            elif not city.strip():

                st.warning(
                    "⚠ Please enter City."
                )

            elif not occupation.strip():

                st.warning(
                    "⚠ Please enter Occupation."
                )

            else:

                st.session_state.account = {

                    "full_name": full_name,

                    "email": email.strip().lower(),

                    "password": password,

                    "age": age,

                    "gender": gender,

                    "city": city,

                    "occupation": occupation

                }


                st.session_state.reg_step = 2
                st.rerun()

# =====================================================
# STEP 2
# HEALTH PROFILE
# =====================================================


if st.session_state.reg_step == 2:

    left,right = st.columns(2)
    with left:

        height_cm = st.number_input(
            "📏 Height (cm)",
            value=170
        )

        weight_kg = st.number_input(
            "⚖ Weight (kg)",
            value=70
        )
        
        bmi = round(
            weight_kg /
            ((height_cm / 100) ** 2),
            2
        )

    with right:

        daily_calorie_limit = st.number_input(
            "🔥 Daily Calories",
            value=2000
        )

        # has_diabetes = st.checkbox(
        #     "Diabetes"
        # )

        # has_hypertension = st.checkbox(
        #     "Hypertension"
        # )

        # has_heart_disease = st.checkbox(
        #     "Heart Disease"
        # )

        # has_obesity = st.checkbox(
        #     "Obesity"
        # )

        allergies = st.text_area(
            "🥜 Allergies"
        )

        dietary_restrictions = st.text_area(
            "🥗 Dietary Restrictions"
        )

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            has_diabetes = st.checkbox("🩸 Diabetes")

        with c2:
            has_hypertension = st.checkbox("💓 Hypertension")

        with c3:
            has_heart_disease = st.checkbox("❤️ Heart Disease")

        with c4:
            has_obesity = st.checkbox("⚖ Obesity")


        col1, col2 = st.columns(2)

        with col1:

            if st.button("⬅ Back",key="step2_back"):

                st.session_state.reg_step = 1
                st.rerun()

        with col2:

            if st.button("Next ➜", key="step2_next"):

                st.session_state.phr = {

                    "height_cm": height_cm,

                    "weight_kg": weight_kg,

                    "bmi": bmi,

                    "has_diabetes": has_diabetes,

                    "has_hypertension": has_hypertension,

                    "has_heart_disease": has_heart_disease,

                    "has_obesity": has_obesity,

                    "allergies": allergies,

                    "dietary_restrictions":
                    dietary_restrictions,

                    "daily_calorie_limit":
                    daily_calorie_limit
                }

                st.session_state.reg_step = 3
                st.rerun()

# =====================================================
# STEP 3
# FOOD PREFERENCES
# =====================================================
if st.session_state.reg_step == 3:

    left,right = st.columns(2)

    with left:

        income_range = st.selectbox(
            "Income Range",
            ["Low", "Middle", "High"]
        )

        monthly_food_budget = st.number_input(
            "Monthly Food Budget",
            value=1000
        )

        lifestyle_type = st.selectbox(
            "Lifestyle",
            [
                "Active",
                "Moderate",
                "Sedentary"
            ]
        )

        personality_type = st.selectbox(
            "Personality",
            [
                "Introvert",
                "Extrovert",
                "Ambivert"
            ]
        )

    with right:
        favorite_cuisine = st.selectbox(
            "Favorite Cuisine",
            [
                "Pakistani",
                "Chinese",
                "Italian",
                "Fast Food",
                "BBQ",
                "Continental",
                "Asian",
                "Middle Eastern"
            ]
        )

        favorite_taste = st.selectbox(
            "Favorite Taste",
            [
                "Spicy",
                "Sweet",
                "Savory",
                "Tangy",
                "Mild"
            ]
        )

        meal_preference = st.selectbox(
            "Meal Preference",
            [
                "Breakfast",
                "Lunch",
                "Dinner",
                "Snacks",
                "Any"
            ]
        )

        food_temp_preference = st.selectbox(
            "Food Temperature",
            [
                "Hot",
                "Cold",
                "Any"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("⬅ Back", key="step3_back"):

                st.session_state.reg_step = 2
                st.rerun()

        with col2:

            if st.button("Next ➜", key="step3_next"):

                st.session_state.profile = {

                    "income_range":
                    income_range,

                    "monthly_food_budget":
                    monthly_food_budget,

                    "lifestyle_type":
                    lifestyle_type,

                    "personality_type":
                    personality_type,

                    "favorite_cuisine":
                    favorite_cuisine,

                    "favorite_taste":
                    favorite_taste,

                    "meal_preference":
                    meal_preference,

                    "food_temp_preference":
                    food_temp_preference
                }

                st.session_state.reg_step = 4
                st.rerun()

# =====================================================
# STEP 4
# REVIEW & REGISTER
# =====================================================

if st.session_state.reg_step == 4:

    st.subheader("✅ Review Your Information")

    account = st.session_state.account
    phr = st.session_state.phr
    profile = st.session_state.profile

    # ==========================================
    # ACCOUNT INFORMATION
    # ==========================================

    st.markdown("## 👤 Account Information")

    col1, col2 = st.columns(2)

    with col1:

        st.info(f"""
        **Full Name**

        {account["full_name"]}

        ---

        **Email**

        {account["email"]}

        ---

        **Age**

        {account["age"]}

        ---

        **Gender**

        {account["gender"]}
        """)

    with col2:

        st.info(f"""
        **City**

        {account["city"]}

        ---

        **Occupation**

        {account["occupation"]}

        ---

        **Password**

        ********
        """)

    st.markdown("---")

    # ==========================================
    # HEALTH PROFILE
    # ==========================================

    st.markdown("## ❤️ Health Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.info(f"""
        **Height**

        {phr["height_cm"]} cm

        ---

        **Weight**

        {phr["weight_kg"]} kg

        ---

        **BMI**

        {phr["bmi"]}

        ---

        **Daily Calories**

        {phr["daily_calorie_limit"]} kcal
        """)

    with col2:

        st.info(f"""
        **Diabetes**

        {"✅ Yes" if phr["has_diabetes"] else "❌ No"}

        ---

        **Hypertension**

        {"✅ Yes" if phr["has_hypertension"] else "❌ No"}

        ---

        **Heart Disease**

        {"✅ Yes" if phr["has_heart_disease"] else "❌ No"}

        ---

        **Obesity**

        {"✅ Yes" if phr["has_obesity"] else "❌ No"}
        """)

    st.markdown("### 🥜 Allergies")

    if phr["allergies"]:
        st.success(phr["allergies"])
    else:
        st.info("None")

    st.markdown("### 🥗 Dietary Restrictions")

    if phr["dietary_restrictions"]:
        st.success(phr["dietary_restrictions"])
    else:
        st.info("None")

    st.markdown("---")

    # ==========================================
    # FOOD PREFERENCES
    # ==========================================

    st.markdown("## 🍽 Food Preferences")

    col1, col2 = st.columns(2)

    with col1:

        st.info(f"""
        **Income Range**

        {profile["income_range"]}

        ---

        **Monthly Food Budget**

        PKR {profile["monthly_food_budget"]}

        ---

        **Lifestyle**

        {profile["lifestyle_type"]}

        ---

        **Personality**

        {profile["personality_type"]}
        """)

    with col2:

        st.info(f"""
        **Favorite Cuisine**

        {profile["favorite_cuisine"]}

        ---

        **Favorite Taste**

        {profile["favorite_taste"]}

        ---

        **Meal Preference**

        {profile["meal_preference"]}

        ---

        **Food Temperature**

        {profile["food_temp_preference"]}
        """)

    st.markdown("---")

    # ==========================================
    # BUTTONS
    # ==========================================

    left, center, right = st.columns([1,4,1])

    with left:

        back = st.button("⬅ Back", key="step4_back")

    with right:

        register = st.button("✅ Register", key="step4_register")

    if back:

        st.session_state.reg_step = 3
        st.rerun()

    if register:

        # =========================
        # EMAIL CHECK
        # =========================

        email = account["email"]

        existing = fetch_data(
            f"""
            SELECT *
            FROM Users
            WHERE LOWER(email)=LOWER('{email}')
            """
        )

        if len(existing) > 0:

            st.error("Email already exists.")

        else:

            user_id = get_next_user_id()

            password_hash = bcrypt.hashpw(
                account["password"].encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")

            # -----------------------------
            # USERS
            # -----------------------------

            execute_query(
                """
                INSERT INTO Users
                (
                    user_id,
                    full_name,
                    email,
                    age,
                    gender,
                    city,
                    occupation,
                    password_hash
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    user_id,
                    account["full_name"],
                    email,
                    account["age"],
                    account["gender"],
                    account["city"],
                    account["occupation"],
                    password_hash
                )
            )

            # -----------------------------
            # USER_PHR
            # -----------------------------

            execute_query(
                """
                INSERT INTO User_PHR
                (
                    user_id,
                    height_cm,
                    weight_kg,
                    bmi,
                    has_diabetes,
                    has_hypertension,
                    has_heart_disease,
                    has_obesity,
                    allergies,
                    dietary_restrictions,
                    daily_calorie_limit
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    user_id,
                    phr["height_cm"],
                    phr["weight_kg"],
                    phr["bmi"],
                    phr["has_diabetes"],
                    phr["has_hypertension"],
                    phr["has_heart_disease"],
                    phr["has_obesity"],
                    phr["allergies"],
                    phr["dietary_restrictions"],
                    phr["daily_calorie_limit"]
                )
            )

            # -----------------------------
            # USER_PROFILE
            # -----------------------------

            execute_query(
                """
                INSERT INTO User_Profile
                (
                    user_id,
                    income_range,
                    monthly_food_budget,
                    lifestyle_type,
                    personality_type,
                    favorite_cuisine,
                    favorite_taste,
                    meal_preference,
                    food_temp_preference
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    user_id,
                    profile["income_range"],
                    profile["monthly_food_budget"],
                    profile["lifestyle_type"],
                    profile["personality_type"],
                    profile["favorite_cuisine"],
                    profile["favorite_taste"],
                    profile["meal_preference"],
                    profile["food_temp_preference"]
                )
            )

            st.success("""
🎉 Registration Completed Successfully!

Welcome to the AI Emotion-Aware Food Recommendation System.

You can now login and enjoy:

✅ Emotion Detection

✅ Personalized Food Recommendation

✅ Nearby Restaurant Recommendation

✅ Hybrid AI Recommendation Engine
""")

            st.balloons()





























# if St.session_state.reg_step == 4:

#     account = st.session_state.account
#     phr = st.session_state.phr
#     profile = st.session_state.profile

#     st.subheader("📋 Review Your Information")

#     # ==========================================
#     # ACCOUNT INFORMATION
#     # ==========================================

#     st.markdown("""
#     <div class="custom-card">
#     <h3>👤 Account Information</h3>
#     </div>
#     """, unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     with col1:

#         st.write(f"**Full Name:** {account['full_name']}")
#         st.write(f"**Email:** {account['email']}")
#         st.write(f"**Age:** {account['age']} Years")
#         st.write(f"**Gender:** {account['gender']}")

#     with col2:

#         st.write(f"**City:** {account['city']}")
#         st.write(f"**Occupation:** {account['occupation']}")

#     st.markdown("---")

#     # ==========================================
#     # HEALTH PROFILE
#     # ==========================================

#     st.markdown("""
#     <div class="custom-card">
#     <h3>❤️ Health Profile</h3>
#     </div>
#     """, unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     with col1:

#         st.write(f"**Height:** {phr['height_cm']} cm")
#         st.write(f"**Weight:** {phr['weight_kg']} kg")
#         st.write(f"**BMI:** {phr['bmi']}")
#         st.write(f"**Daily Calories:** {phr['daily_calorie_limit']} kcal")

#     with col2:

#         st.write(
#             f"**Diabetes:** {'✅ Yes' if phr['has_diabetes'] else '❌ No'}"
#         )

#         st.write(
#             f"**Hypertension:** {'✅ Yes' if phr['has_hypertension'] else '❌ No'}"
#         )

#         st.write(
#             f"**Heart Disease:** {'✅ Yes' if phr['has_heart_disease'] else '❌ No'}"
#         )

#         st.write(
#             f"**Obesity:** {'✅ Yes' if phr['has_obesity'] else '❌ No'}"
#         )

#     st.write(
#         f"**Allergies:** {phr['allergies'] if phr['allergies'] else 'None'}"
#     )

#     st.write(
#         f"**Dietary Restrictions:** {phr['dietary_restrictions'] if phr['dietary_restrictions'] else 'None'}"
#     )

#     st.markdown("---")

#     # ==========================================
#     # FOOD PREFERENCES
#     # ==========================================

#     st.markdown("""
#     <div class="custom-card">
#     <h3>🍽 Food Preferences</h3>
#     </div>
#     """, unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     with col1:

#         st.write(f"**Income Range:** {profile['income_range']}")
#         st.write(f"**Monthly Budget:** PKR {profile['monthly_food_budget']}")
#         st.write(f"**Lifestyle:** {profile['lifestyle_type']}")
#         st.write(f"**Personality:** {profile['personality_type']}")

#     with col2:

#         st.write(f"**Favorite Cuisine:** {profile['favorite_cuisine']}")
#         st.write(f"**Favorite Taste:** {profile['favorite_taste']}")
#         st.write(f"**Meal Preference:** {profile['meal_preference']}")
#         st.write(f"**Food Temperature:** {profile['food_temp_preference']}")

#     st.markdown("---")

#     # ==========================================
#     # NAVIGATION BUTTONS
#     # ==========================================

#     left, middle, right = st.columns([1, 5, 1])

#     with left:

#         back = st.button("⬅ Back")

#     with right:

#         register = st.button("✅ Register")

#     if back:

#         st.session_state.reg_step = 3
#         st.rerun()

#     # ==========================================
#     # REGISTER USER
#     # ==========================================

#     if register:

#         # =========================
#         # EMAIL CHECK
#         # =========================

#         email = account["email"]

#         existing = fetch_data(
#             f"""
#             SELECT *
#             FROM Users
#             WHERE LOWER(email)=LOWER('{email}')
#             """
#         )

#         if len(existing) > 0:

#             st.error("Email already exists.")

#         else:

#             user_id = get_next_user_id()

#             password_hash = bcrypt.hashpw(
#                 account["password"].encode("utf-8"),
#                 bcrypt.gensalt()
#             ).decode("utf-8")

#             # =========================
#             # USERS
#             # =========================

#             execute_query(
#                 """
#                 INSERT INTO Users
#                 (
#                     user_id,
#                     full_name,
#                     email,
#                     age,
#                     gender,
#                     city,
#                     occupation,
#                     password_hash
#                 )
#                 VALUES
#                 (%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     user_id,
#                     account["full_name"],
#                     email,
#                     account["age"],
#                     account["gender"],
#                     account["city"],
#                     account["occupation"],
#                     password_hash
#                 )
#             )

#             # =========================
#             # USER_PHR
#             # =========================

#             execute_query(
#                 """
#                 INSERT INTO User_PHR
#                 (
#                     user_id,
#                     height_cm,
#                     weight_kg,
#                     bmi,
#                     has_diabetes,
#                     has_hypertension,
#                     has_heart_disease,
#                     has_obesity,
#                     allergies,
#                     dietary_restrictions,
#                     daily_calorie_limit
#                 )
#                 VALUES
#                 (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     user_id,
#                     phr["height_cm"],
#                     phr["weight_kg"],
#                     phr["bmi"],
#                     phr["has_diabetes"],
#                     phr["has_hypertension"],
#                     phr["has_heart_disease"],
#                     phr["has_obesity"],
#                     phr["allergies"],
#                     phr["dietary_restrictions"],
#                     phr["daily_calorie_limit"]
#                 )
#             )

#             # =========================
#             # USER_PROFILE
#             # =========================

#             execute_query(
#                 """
#                 INSERT INTO User_Profile
#                 (
#                     user_id,
#                     income_range,
#                     monthly_food_budget,
#                     lifestyle_type,
#                     personality_type,
#                     favorite_cuisine,
#                     favorite_taste,
#                     meal_preference,
#                     food_temp_preference
#                 )
#                 VALUES
#                 (%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     user_id,
#                     profile["income_range"],
#                     profile["monthly_food_budget"],
#                     profile["lifestyle_type"],
#                     profile["personality_type"],
#                     profile["favorite_cuisine"],
#                     profile["favorite_taste"],
#                     profile["meal_preference"],
#                     profile["food_temp_preference"]
#                 )
#             )

#             st.success("""
# 🎉 Registration Completed Successfully!

# Welcome to the AI Emotion-Aware Food Recommendation System.

# You can now enjoy:

# ✅ Emotion Detection

# ✅ Personalized Food Recommendation

# ✅ Nearby Restaurant Recommendation

# ✅ AI Hybrid Recommendation Engine
# """)

#             st.balloons()

#             if st.button("🔐 Go To Login"):

#                 st.session_state.clear()

#                 st.switch_page("pages/1_Login.py")