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

st.title("📝 User Registration")

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

    st.subheader("Step 1: Account Information")

    full_name = st.text_input("Full Name")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password"
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    city = st.text_input("City")

    occupation = st.text_input(
        "Occupation"
    )

    if st.button("Next ➜"):

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

elif st.session_state.reg_step == 2:

    st.subheader(
        "Step 2: Health Profile"
    )

    height_cm = st.number_input(
        "Height (cm)",
        value=170
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        value=70
    )

    bmi = round(
        weight_kg /
        ((height_cm / 100) ** 2),
        2
    )

    st.info(f"BMI: {bmi}")

    has_diabetes = st.checkbox(
        "Diabetes"
    )

    has_hypertension = st.checkbox(
        "Hypertension"
    )

    has_heart_disease = st.checkbox(
        "Heart Disease"
    )

    has_obesity = st.checkbox(
        "Obesity"
    )

    allergies = st.text_area(
        "Allergies"
    )

    dietary_restrictions = st.text_area(
        "Dietary Restrictions"
    )

    daily_calorie_limit = st.number_input(
        "Daily Calorie Limit",
        value=2000
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("⬅ Back"):

            st.session_state.reg_step = 1
            st.rerun()

    with col2:

        if st.button("Next ➜"):

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

elif st.session_state.reg_step == 3:

    st.subheader(
        "Step 3: Food Preferences"
    )

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

        if st.button("⬅ Back"):

            st.session_state.reg_step = 2
            st.rerun()

    with col2:

        if st.button("Next ➜"):

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

elif st.session_state.reg_step == 4:

    st.subheader(
        "Step 4: Review Information"
    )

    st.write(
        st.session_state.account
    )

    st.write(
        st.session_state.phr
    )

    st.write(
        st.session_state.profile
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("⬅ Back"):

            st.session_state.reg_step = 3
            st.rerun()

    with col2:

        if st.button("Register"):

            # =========================
            # EMAIL CHECK
            # =========================

            email = st.session_state.account[
                "email"
            ]

            existing = fetch_data(
                f"""
                SELECT *
                FROM Users
                WHERE LOWER(email)=LOWER('{email}')
                """
            )

            if len(existing) > 0:

                st.error(
                    "Email already exists."
                )

            else:

                user_id = get_next_user_id()

                password_hash = bcrypt.hashpw(
                    st.session_state.account[
                        "password"
                    ].encode("utf-8"),
                    bcrypt.gensalt()
                ).decode("utf-8")

                # =========================
                # USERS
                # =========================

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
                        st.session_state.account[
                            "full_name"
                        ],
                        email,
                        st.session_state.account[
                            "age"
                        ],
                        st.session_state.account[
                            "gender"
                        ],
                        st.session_state.account[
                            "city"
                        ],
                        st.session_state.account[
                            "occupation"
                        ],
                        password_hash
                    )
                )

                # =========================
                # USER_PHR
                # =========================

                p = st.session_state.phr

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
                        p["height_cm"],
                        p["weight_kg"],
                        p["bmi"],
                        p["has_diabetes"],
                        p["has_hypertension"],
                        p["has_heart_disease"],
                        p["has_obesity"],
                        p["allergies"],
                        p["dietary_restrictions"],
                        p["daily_calorie_limit"]
                    )
                )

                # =========================
                # USER_PROFILE
                # =========================

                pr = st.session_state.profile

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
                        pr["income_range"],
                        pr["monthly_food_budget"],
                        pr["lifestyle_type"],
                        pr["personality_type"],
                        pr["favorite_cuisine"],
                        pr["favorite_taste"],
                        pr["meal_preference"],
                        pr["food_temp_preference"]
                    )
                )

                st.success("Registration Successful!")

                st.balloons()

                if st.button("Go to Login"):
                    st.switch_page("1_Login.py")