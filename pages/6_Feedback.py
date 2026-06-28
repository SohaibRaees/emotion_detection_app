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

st.title(
    "⭐ Feedback & Ratings"
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

# =====================================================
# RATING
# =====================================================

rating = st.slider(
    "Rate Recommendation",
    min_value=1,
    max_value=5,
    value=5
)

comment = st.text_area(
    "Comment"
)

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
    "Submit Feedback"
):

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
            comment,
            emotion
        )
    )

    # =========================================
    # SAVE USER INTERACTION
    # =========================================

    interaction_id_df = fetch_data(
    """
    SELECT COALESCE(MAX(interaction_id),0)+1 AS next_id
    FROM User_Interactions
    """
    )

    interaction_id = int(
        interaction_id_df.iloc[0]["next_id"]
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

    st.success(
        "Feedback submitted successfully."
    )

    st.balloons()

# =====================================================
# FEEDBACK HISTORY
# =====================================================

st.markdown("---")

st.subheader(
    "📜 Your Previous Feedback"
)

feedback_history = fetch_data(
    f"""
    SELECT
        f.food_name,
        r.restaurant_name,
        fb.rating,
        fb.comment,
        fb.created_at
    FROM Feedback fb
    JOIN Foods f
        ON fb.food_id=f.food_id
    JOIN Restaurants r
        ON fb.restaurant_id=r.restaurant_id
    WHERE fb.user_id={user_id}
    ORDER BY fb.created_at DESC
    LIMIT 20
    """
)

if not feedback_history.empty:

    st.dataframe(
        feedback_history,
        use_container_width=True
    )

else:

    st.info(
        "No feedback submitted yet."
    )