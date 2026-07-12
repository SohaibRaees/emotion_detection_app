# =====================================================
# FEEDBACK PAGE
# =====================================================

import streamlit as st

from database.db_connection import (
    fetch_data,
    execute_insert
)

# =====================================================
# LOGIN CHECK
# =====================================================

if (
    "user_id" not in st.session_state
):
    st.warning(
        "Please login first."
    )
    st.stop()

# =====================================================
# PAGE TITLE
# =====================================================

from utils.ui import (
    load_ui,
    page_header
)

load_ui("Feedback Page")

page_header(
    "Feedback Page",
    "⭐"
)


user_id = st.session_state.user_id

emotion = st.session_state.get(
    "emotion",
    "Unknown"
)

# =====================================================
# LATEST RECOMMENDATION SESSION
# =====================================================

history_df = fetch_data(
    f"""
    SELECT *
    FROM Recommendation_History
    WHERE user_id={user_id}
    ORDER BY history_id DESC
    LIMIT 1
    """
)

if history_df.empty:

    st.warning(
        "No recommendation session found."
    )

    st.stop()

history_id = int(
    history_df.iloc[0]["history_id"]
)

# =====================================================
# RECOMMENDED FOODS
# =====================================================

foods_df = fetch_data(
    f"""
    SELECT
        ri.food_id,
        f.food_name
    FROM Recommendation_Items ri
    JOIN Foods f
        ON ri.food_id = f.food_id
    WHERE ri.history_id={history_id}
    ORDER BY ri.rank_position
    """
)

if foods_df.empty:

    st.warning(
        "No recommended foods found."
    )

    st.stop()

# =====================================================
# FEEDBACK DASHBOARD
# =====================================================

st.markdown("---")

col1,col2,col3=st.columns(3)

with col1:

    st.metric(
        "😊 Current Emotion",
        emotion
    )

with col2:

    st.metric(
        "🍽 Recommended Foods",
        len(foods_df) if 'foods_df' in locals() else "-"
    )

with col3:

    st.metric(
        "⭐ Your Rating",
        "Pending"
    )

st.markdown("---")

# =====================================================
# FOOD SELECTION
# =====================================================

food_options = {
    row["food_name"]: row["food_id"]
    for _, row in foods_df.iterrows()
}

selected_food = st.selectbox(
    "Select Food",
    list(food_options.keys())
)

food_id = food_options[
    selected_food
]

# =====================================================
# RESTAURANT SELECTION
# =====================================================

restaurant_df = fetch_data(
    f"""
    SELECT DISTINCT
        r.restaurant_id,
        r.restaurant_name
    FROM Restaurants r
    JOIN Restaurant_Foods rf
        ON r.restaurant_id = rf.restaurant_id
    WHERE rf.food_id={food_id}
    """
)

restaurant_options = {
    row["restaurant_name"]:
    row["restaurant_id"]
    for _, row in restaurant_df.iterrows()
}

selected_restaurant = st.selectbox(
    "Select Restaurant",
    list(restaurant_options.keys())
)

restaurant_id = restaurant_options[
    selected_restaurant
]


st.markdown("---")

left,right=st.columns(2)

with left:

    st.markdown(f"""
<div class="custom-card">

<h3>🍔 Selected Food</h3>

<h2>{selected_food}</h2>

<p>
Recommended specifically
for your detected emotion.
</p>

</div>
""",unsafe_allow_html=True)

with right:

    st.markdown(f"""
<div class="custom-card">

<h3>🍴 Restaurant</h3>

<h2>{selected_restaurant}</h2>

<p>

Best nearby restaurant
based on TOPSIS ranking.

</p>

</div>
""",unsafe_allow_html=True)



#EMOTION CARD
st.markdown(f"""
<div class="custom-card">

<h3>😊 Emotion During Recommendation</h3>

<h1 style="color:#16A34A">

{emotion}

</h1>

</div>
""",
unsafe_allow_html=True)

# =====================================================
# RATING
# =====================================================

st.subheader("⭐ Rate Your Experience")

rating = st.radio(

    "",

    [

        1,

        2,

        3,

        4,

        5

    ],

    horizontal=True,

    format_func=lambda x:

    "⭐"*x

)


comment = st.text_area(

    "💬 Share Your Experience",

    placeholder=

    """
Tell us:

• Was the recommendation useful?

• Was the food delicious?

• Was the restaurant good?

• Would you order again?

""",

    height=180

)




#Tags
st.subheader("👍 Quick Review")

col1,col2,col3=st.columns(3)

with col1:

    tasty=st.checkbox("😋 Tasty")

    healthy=st.checkbox("🥗 Healthy")

with col2:

    affordable=st.checkbox("💰 Affordable")

    fast=st.checkbox("⚡ Fast Delivery")

with col3:

    recommend=st.checkbox("❤️ Would Recommend")

    clean=st.checkbox("✨ Clean Restaurant")

# =====================================================
# SUBMIT
# =====================================================
# =====================================================
# SUBMIT
# =====================================================

feedback_id_df = fetch_data(
    """
    SELECT COALESCE(MAX(feedback_id),0)+1 AS next_id
    FROM Feedback
    """
)

feedback_id = int(
    feedback_id_df.iloc[0]["next_id"]
)

if st.button(
    "⭐ Submit Feedback",
    use_container_width=True,
    key="submit_feedback"
):

    # ==========================================
    # BUILD TAG LIST
    # ==========================================

    tags = []

    if tasty:
        tags.append("Tasty")

    if healthy:
        tags.append("Healthy")

    if affordable:
        tags.append("Affordable")

    if fast:
        tags.append("Fast Delivery")

    if recommend:
        tags.append("Would Recommend")

    if clean:
        tags.append("Clean")

    final_comment = comment.strip()

    if tags:

        if final_comment:

            final_comment += "\n\n"

        final_comment += "Tags: " + ", ".join(tags)

    # ==========================================
    # SAVE FEEDBACK
    # ==========================================

    execute_insert(
        """
        INSERT INTO Feedback
        (
            feedback_id,
            user_id,
            food_id,
            restaurant_id,
            rating,
            comment,
            emotion_at_time
        )
        VALUES
        (%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            feedback_id,
            user_id,
            food_id,
            restaurant_id,
            rating,
            final_comment,
            emotion
        )
    )

    # ==========================================
    # USER INTERACTION
    # ==========================================

    interaction_df = fetch_data(
        """
        SELECT COALESCE(MAX(interaction_id),0)+1 AS next_id
        FROM User_Interactions
        """
    )

    interaction_id = int(
        interaction_df.iloc[0]["next_id"]
    )

    execute_insert(
        """
        INSERT INTO User_Interactions
        (
            interaction_id,
            user_id,
            food_id,
            interaction_type,
            interaction_weight,
            emotion_at_time
        )
        VALUES
        (%s,%s,%s,%s,%s,%s)
        """,
        (
            interaction_id,
            user_id,
            food_id,
            "Feedback",
            rating,
            emotion
        )
    )

    st.success("""
🎉 Thank You!

Your feedback has been recorded successfully.

Your review will help improve future AI-powered recommendations.
""")

    st.balloons()

    st.markdown(
        """
<div class="custom-card">

<center>

<h2>❤️ Thank You</h2>

<p>

Your feedback makes our AI smarter.

We appreciate your contribution.

</p>

</center>

</div>
""",
        unsafe_allow_html=True
    )

# =====================================================
# FEEDBACK HISTORY
# =====================================================

# st.markdown("---")

# st.subheader(
#     "📜 Your Previous Feedback"
# )

# feedback_history = fetch_data(
#     f"""
#     SELECT
#         f.food_name,
#         r.restaurant_name,
#         fb.rating,
#         fb.comment,
#         fb.created_at
#     FROM Feedback fb
#     JOIN Foods f
#         ON fb.food_id=f.food_id
#     JOIN Restaurants r
#         ON fb.restaurant_id=r.restaurant_id
#     WHERE fb.user_id={user_id}
#     ORDER BY fb.created_at DESC
#     LIMIT 20
#     """
# )

# if not feedback_history.empty:

#     for _, row in feedback_history.iterrows():

#         stars = "⭐"*int(row["rating"])

#         st.markdown(f"""
#             <div class="custom-card">

#             <h3>

#             🍔 {row['food_name']}

#             </h3>

#             <b>

#             🍴 {row['restaurant_name']}

#             </b>

#             <br><br>

#             {stars}

#             <br><br>

#             💬 {row['comment']}

#             <br><br>

#             <small>

#             📅 {row['created_at']}

#             </small>

#             </div>

#             """,
#             unsafe_allow_html=True)

# else:

#     st.info(
#         "No feedback submitted yet."
#     )