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

# st.title(
#     "🍽 Personalized Food Recommendations"
# )
from utils.ui import (
    load_ui,
    page_intro
)

load_ui("Food Recommendations")

page_intro(

    "Food Recommendation",

    "Hybrid AI engine generating personalized meal suggestions.",

    "🍔"

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
# DISPLAY COLUMNS
# =====================================================

display_columns = [
    "food_name",
    "cuisine_type",
    "base_price",
    "calories_per_serving",
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


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Foods Found",
        len(recommendations)
    )

with col2:
    st.metric(
        "Emotion",
        emotion
    )

with col3:
    st.metric(
        "Best Score",
        f"{recommendations.iloc[0]['final_score']:.2f}"
    )

with col4:
    st.metric(
        "Top Cuisine",
        recommendations.iloc[0]["cuisine_type"]
    )

st.markdown("---")

# =====================================================
# DISPLAY TABLE
# =====================================================

st.subheader("🏆 Personalized Food Recommendations")

with st.expander("📊 View Recommendation Scores (Optional)"):

    st.dataframe(
        recommendations[
            available_columns
        ],
        use_container_width=True
    )

# =====================================================
# PREMIUM FOOD CARDS
# =====================================================

medals = [
    "🥇",
    "🥈",
    "🥉",
    "4️⃣",
    "5️⃣",
    "6️⃣",
    "7️⃣",
    "8️⃣",
    "9️⃣",
    "🔟"
]

for rank, (_, row) in enumerate(
    recommendations.iterrows(),
    start=1
):

    medal = medals[rank-1]

    score = float(row["final_score"])

    score_percent = min(score * 100, 100)

    badges = []

    badges.append(f"🍽 {row['cuisine_type']}")

    if "taste_profile" in row:
        badges.append(f"😋 {row['taste_profile']}")

    badges.append("❤️ Emotion Match")

    if row["calories_per_serving"] <= 350:
        badges.append("🥗 Healthy")

    if row["base_price"] <= 700:
        badges.append("💰 Budget Friendly")

    badge_html = "".join(
        [
            f"<span class='food-badge'>{b}</span>"
            for b in badges
        ]
    )

    st.markdown(
        f"""
<div class="food-card">

<div style="
display:flex;
justify-content:space-between;
align-items:center;">

<div>

<div class="food-title">

{medal} {row['food_name']}

</div>

<div style="color:#64748B;">

Recommended because it matches your current emotion and profile.

</div>

</div>

<div class="food-score">

{score_percent:.0f}%

</div>

</div>

<br>

<div style="
display:flex;
justify-content:space-between;
flex-wrap:wrap;
font-size:15px;">

<div>

💰 <b>PKR {row['base_price']:.0f}</b>

</div>

<div>

🔥 <b>{row['calories_per_serving']} kcal</b>

</div>

<div>

🍗 <b>{row['protein_g']} g Protein</b>

</div>

<div>

🥖 <b>{row['carbs_g']} g Carbs</b>

</div>

<div>

🧈 <b>{row['fat_g']} g Fat</b>

</div>

</div>

<br>

<div class="food-progress">

<div

class="food-progress-fill"

style="width:{score_percent}%">

</div>

</div>

<br>

{badge_html}

</div>

""",
        unsafe_allow_html=True
    )
# =====================================================
# SAVE RECOMMENDATION SESSION
# =====================================================

left, right = st.columns(2)
with left:

    save_clicked = st.button(
        "💾 Save Recommendation Session",
        use_container_width=True
    )

with right:

    continue_clicked = st.button(
        "🍴 View Nearby Restaurants",
        use_container_width=True
    )

if save_clicked:

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

if continue_clicked:

    st.switch_page(
        "pages/5_Restaurant_Recommendation.py"
    )