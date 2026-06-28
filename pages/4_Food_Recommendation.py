# =====================================================
# FOOD RECOMMENDATION PAGE
# =====================================================

import streamlit as st

from recommendation.hybrid_recommender import (
    generate_recommendations
)

from database.db_connection import (
    fetch_data,
    execute_insert,
    execute_query
)

# =====================================================
# LOGIN CHECK
# =====================================================

if (
    "user_id" not in st.session_state
    or
    st.session_state.user_id is None
):
    st.warning("Please login first.")
    st.stop()

# =====================================================
# EMOTION CHECK
# =====================================================

if (
    "emotion" not in st.session_state
):
    st.warning(
        "Please detect emotion first."
    )
    st.stop()

# =====================================================
# USER INFO
# =====================================================

user_id = st.session_state.user_id

emotion = st.session_state.emotion

confidence = st.session_state.get(
    "confidence",
    0
)

emotion_id = st.session_state.get(
    "emotion_id",
    None
)

# =====================================================
# PAGE HEADER
# =====================================================

st.title(
    "🍽 Personalized Food Recommendations"
)

st.success(
    f"Current Emotion: {emotion}"
)

st.info(
    f"Confidence: {confidence*100:.2f}%"
)

# =====================================================
# GENERATE RECOMMENDATIONS
# =====================================================

with st.spinner(
    "Generating recommendations..."
):

    recommendations = (
        generate_recommendations(
            user_id=user_id,
            emotion=emotion
        )
    )

# =====================================================
# DISPLAY TABLE
# =====================================================

st.subheader(
    "🏆 Top Food Recommendations"
)

display_columns = [
    "food_name",
    "emotion_score",
    "pref_score",
    "user_cf_score",
    "item_cf_score",
    "feedback_score",
    "final_score"
]

available_columns = [
    col
    for col in display_columns
    if col in recommendations.columns
]

st.dataframe(
    recommendations[
        available_columns
    ],
    use_container_width=True
)

# =====================================================
# FOOD CARDS
# =====================================================

st.markdown("---")

for rank, (_, row) in enumerate(
    recommendations.iterrows(),
    start=1
):

    st.container()

    st.markdown(
        f"""
### #{rank} {row['food_name']}

**Cuisine:** {row['cuisine_type']}

**Price:** PKR {row['base_price']}

**Calories:** {row['calories_per_serving']}

**Final Score:** {row['final_score']:.4f}

---
"""
    )

# =====================================================
# SAVE RECOMMENDATION SESSION
# =====================================================

if st.button(
    "💾 Save Recommendation Session"
):

    if emotion_id is None:

        st.error(
            "Emotion session not found. Please save emotion first."
        )

    else:

        # =========================================
        # USER PROFILE
        # =========================================

        profile = fetch_data(
            f"""
            SELECT *
            FROM User_Profile
            WHERE user_id={user_id}
            """
        )

        budget = float(
            profile.iloc[0][
                "monthly_food_budget"
            ]
        )

        # =========================================
        # INSERT HISTORY
        # =========================================

        history_id = execute_insert(
            """
            INSERT INTO Recommendation_History
            (
                user_id,
                emotion_id,
                emotion_at_time,
                emotion_confidence,
                budget_at_time,
                method_used
            )
            VALUES
            (%s,%s,%s,%s,%s,%s)
            """,
            (
                user_id,
                emotion_id,
                emotion,
                confidence,
                budget,
                "Hybrid Recommender"
            )
        )

        # =========================================
        # INSERT TOP 10 ITEMS
        # =========================================

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            execute_query(
                """
                INSERT INTO Recommendation_Items
                (
                    history_id,
                    food_id,
                    final_score,
                    rank_position
                )
                VALUES
                (%s,%s,%s,%s)
                """,
                (
                    history_id,
                    int(row["food_id"]),
                    float(row["final_score"]),
                    rank
                )
            )

        st.success(
            "Recommendation session saved successfully."
        )

# =====================================================
# CONTINUE
# =====================================================

st.markdown("---")

if st.button(
    "➡ Continue To Restaurant Recommendation"
):

    st.switch_page(
        "pages/5_Restaurant_Recommendation.py"
    )