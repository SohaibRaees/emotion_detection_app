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
# HYBRID RECOMMENDER ENGINE
# =====================================================

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from database.db_connection import fetch_data


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    foods = fetch_data(
        "SELECT * FROM Foods"
    )

    emotion_mapping = fetch_data(
        "SELECT * FROM Emotion_Food_Mapping"
    )

    feedback = fetch_data(
        "SELECT * FROM Feedback"
    )

    interactions = fetch_data(
        "SELECT * FROM User_Interactions"
    )

    return (
        foods,
        emotion_mapping,
        feedback,
        interactions
    )


# =====================================================
# LOAD USER PROFILE
# =====================================================

def get_user_profile(user_id):

    profile = fetch_data(
        f"""
        SELECT *
        FROM User_Profile
        WHERE user_id={user_id}
        """
    )

    return profile.iloc[0]


# =====================================================
# LOAD USER PHR
# =====================================================

def get_user_phr(user_id):

    phr = fetch_data(
        f"""
        SELECT *
        FROM User_PHR
        WHERE user_id={user_id}
        """
    )

    return phr.iloc[0]


# =====================================================
# HEALTH FILTER
# =====================================================

def apply_health_filter(
    foods,
    user_phr
):

    safe_foods = foods.copy()

    # Diabetes

    if user_phr["has_diabetes"] == 1:

        safe_foods = safe_foods[
            safe_foods["is_high_sugar"] == 0
        ]

    # Heart Disease

    if user_phr["has_heart_disease"] == 1:

        safe_foods = safe_foods[
            safe_foods["fat_g"] <= 15
        ]

    # Obesity

    if user_phr["has_obesity"] == 1:

        calorie_limit = (
            user_phr["daily_calorie_limit"] / 3
        )

        safe_foods = safe_foods[
            safe_foods["calories_per_serving"]
            <= calorie_limit
        ]

    return safe_foods


# =====================================================
# BUDGET FILTER
# =====================================================

def apply_budget_filter(
    foods,
    profile
):

    return foods[
        foods["base_price"]
        <=
        profile["monthly_food_budget"]
    ]


# =====================================================
# PREFERENCE SCORE
# =====================================================

def add_preference_score(
    foods,
    profile
):

    foods = foods.copy()

    foods["pref_score"] = 0.0

    foods.loc[
        foods["cuisine_type"]
        ==
        profile["favorite_cuisine"],
        "pref_score"
    ] += 0.5

    foods.loc[
        foods["taste_profile"]
        ==
        profile["favorite_taste"],
        "pref_score"
    ] += 0.5

    return foods


# =====================================================
# EMOTION SCORE
# =====================================================

def add_emotion_score(
    foods,
    emotion,
    emotion_mapping
):

    foods = foods.merge(
        emotion_mapping[
            emotion_mapping["emotion"]
            ==
            emotion
        ][
            [
                "food_id",
                "relevance_score"
            ]
        ],
        on="food_id",
        how="left"
    )

    foods["emotion_score"] = foods[
        "relevance_score"
    ].fillna(0)

    return foods


# =====================================================
# USER BASED CF
# =====================================================

def user_based_cf(
    user_id,
    interactions
):

    matrix = interactions.pivot_table(
        index="user_id",
        columns="food_id",
        values="interaction_weight"
    ).fillna(0)

    if user_id not in matrix.index:

        return {}

    similarity = cosine_similarity(
        matrix
    )

    sim_df = pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index
    )

    similar_users = sim_df[
        user_id
    ].sort_values(
        ascending=False
    )[1:]

    scores = {}

    for sim_user, sim_score in similar_users.items():

        for food_id in matrix.columns:

            value = matrix.loc[
                sim_user,
                food_id
            ]

            if value > 0:

                scores[food_id] = (
                    scores.get(food_id, 0)
                    +
                    value * sim_score
                )

    return scores


# =====================================================
# ITEM BASED CF
# =====================================================

def build_item_similarity(
    interactions
):

    matrix = interactions.pivot_table(
        index="user_id",
        columns="food_id",
        values="interaction_weight"
    ).fillna(0)

    similarity = cosine_similarity(
        matrix.T
    )

    return pd.DataFrame(
        similarity,
        index=matrix.columns,
        columns=matrix.columns
    )


# =====================================================
# FEEDBACK SCORE
# =====================================================

def build_feedback_dict(
    feedback
):

    if feedback.empty:

        return {}

    return (
        feedback
        .groupby("food_id")
        ["rating"]
        .mean()
        .to_dict()
    )


# =====================================================
# MAIN RECOMMENDER
# =====================================================

def generate_recommendations(
    user_id,
    emotion
):

    (
        foods,
        emotion_mapping,
        feedback,
        interactions
    ) = load_data()

    profile = get_user_profile(
        user_id
    )

    phr = get_user_phr(
        user_id
    )

    # --------------------------------

    safe_foods = apply_health_filter(
        foods,
        phr
    )

    safe_foods = apply_budget_filter(
        safe_foods,
        profile
    )

    safe_foods = add_preference_score(
        safe_foods,
        profile
    )

    safe_foods = add_emotion_score(
        safe_foods,
        emotion,
        emotion_mapping
    )

    # --------------------------------

    cf_scores = user_based_cf(
        user_id,
        interactions
    )

    safe_foods["user_cf_score"] = (
        safe_foods["food_id"]
        .map(cf_scores)
        .fillna(0)
    )

    # --------------------------------

    item_sim_df = build_item_similarity(
        interactions
    )

    safe_foods["item_cf_score"] = (
        safe_foods["food_id"]
        .apply(
            lambda x:
            item_sim_df.loc[x].mean()
            if x in item_sim_df.index
            else 0
        )
    )

    # --------------------------------

    feedback_dict = (
        build_feedback_dict(
            feedback
        )
    )

    safe_foods["feedback_score"] = (
        safe_foods["food_id"]
        .apply(
            lambda x:
            feedback_dict.get(
                x,
                3
            ) / 5
        )
    )

    # --------------------------------
    # FINAL SCORE
    # --------------------------------

    safe_foods["final_score"] = (

        0.30 * safe_foods["emotion_score"]

        +

        0.20 * safe_foods["pref_score"]

        +

        0.20 * safe_foods["user_cf_score"]

        +

        0.10 * safe_foods["item_cf_score"]

        +

        0.20 * safe_foods["feedback_score"]

    )

    recommendations = (
        safe_foods
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(10)
    )

    return recommendations