import folium

from streamlit_folium import (
    st_folium
)

# =====================================================
# RESTAURANT RECOMMENDATION
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from database.db_connection import (
    fetch_data,
    execute_query
)

from utils.location_service import (
    get_user_location,
    add_distance_to_restaurants,
    filter_nearby_restaurants,
    get_city_name
)

# =====================================================
# LOGIN CHECK
# =====================================================

if "user_id" not in st.session_state:
    st.warning("Please login first.")
    st.stop()

user_id = st.session_state.user_id

st.title("🍴 Nearby Restaurant Recommendations")

# =====================================================
# USER LOCATION
# =====================================================

user_lat, user_lon = get_user_location(user_id)

if user_lat is None:

    st.error(
        "Current location unavailable.\n\n"
        "Please login again and allow location permission."
    )

    st.stop()

city = get_city_name(user_lat, user_lon)

st.success("📍 Current Location")
st.info(f"🏙 City: {city}")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latitude",
        f"{user_lat:.6f}"
    )

with col2:
    st.metric(
        "Longitude",
        f"{user_lon:.6f}"
    )

# =====================================================
# LOAD LATEST RECOMMENDATION SESSION
# =====================================================

history = fetch_data(f"""
SELECT *
FROM Recommendation_History
WHERE user_id={user_id}
ORDER BY history_id DESC
LIMIT 1
""")

if history.empty:

    st.warning(
        "Generate food recommendations first."
    )

    st.stop()

history_id = int(
    history.iloc[0]["history_id"]
)

# =====================================================
# LOAD TOP RECOMMENDED FOODS
# =====================================================

recommended_foods = fetch_data(f"""
SELECT

ri.food_id,

ri.final_score,

ri.rank_position,

f.food_name

FROM Recommendation_Items ri

JOIN Foods f

ON ri.food_id=f.food_id

WHERE ri.history_id={history_id}

ORDER BY ri.rank_position
""")

# =====================================================
# LOAD RESTAURANTS
# =====================================================

restaurants = fetch_data("""
SELECT *
FROM Restaurants
WHERE is_active=1
""")

restaurant_foods = fetch_data("""
SELECT *
FROM Restaurant_Foods
WHERE is_available=1
""")

# =====================================================
# RESTAURANTS SERVING RECOMMENDED FOODS
# =====================================================

candidate_restaurants = (
    restaurant_foods
    .merge(
        recommended_foods,
        on="food_id"
    )
    .merge(
        restaurants,
        on="restaurant_id"
    )
)

if candidate_restaurants.empty:

    st.warning(
        "No nearby restaurants found."
    )

    st.stop()

# =====================================================
# DISTANCE CALCULATION
# =====================================================

candidate_restaurants = add_distance_to_restaurants(

    candidate_restaurants,

    user_lat,

    user_lon

)

# =====================================================
# FILTER BY DISTANCE
# =====================================================

MAX_DISTANCE = st.slider(

    "Maximum Distance (KM)",

    50,

    200,

    120

)

candidate_restaurants = filter_nearby_restaurants(

    candidate_restaurants,

    MAX_DISTANCE

)

if candidate_restaurants.empty:

    st.warning(

        f"No restaurants found within {MAX_DISTANCE} KM."

    )

    st.stop()

# =====================================================
# TRAVEL TIME
# =====================================================

AVERAGE_CITY_SPEED = 30

candidate_restaurants["travel_time"] = (

    candidate_restaurants["distance_km"]

    /

    AVERAGE_CITY_SPEED

) * 60

candidate_restaurants["travel_time"] = (

    candidate_restaurants["travel_time"]

    .round()

    .astype(int)

)

# =====================================================
# RESTAURANT AGGREGATION
# =====================================================

restaurant_scores = (
    candidate_restaurants
    .groupby(
        [
            "restaurant_id",
            "restaurant_name",
            "average_rating",
            "average_price_pkr",
            "avg_delivery_time",
            "offers_delivery",
            "distance_km",
            "travel_time"
        ],
        as_index=False
    )
    .agg(
        {
            "final_score": "mean",
            "food_name": "first",
            "food_id": "first",
            "latitude": "first",
            "longitude": "first"
        
        }
    )
)

st.info(

    f"Found {len(restaurant_scores)} nearby restaurants."

)

# =====================================================
# TOPSIS (Enhanced)
# =====================================================

criteria = restaurant_scores[
    [
        "average_rating",
        "final_score",
        "distance_km",
        "average_price_pkr",
        "avg_delivery_time"
    ]
].copy()

# ---------------------------------------------
# Normalize Decision Matrix
# ---------------------------------------------

criteria = criteria / np.sqrt(
    (criteria ** 2).sum(axis=0)
)

# ---------------------------------------------
# Apply Weights
# ---------------------------------------------

weights = np.array([
    0.30,   # Restaurant Rating
    0.25,   # Food Recommendation Score
    0.20,   # Distance
    0.15,   # Price
    0.10    # Delivery Time
])

weighted = criteria * weights

# ---------------------------------------------
# Ideal Best
# ---------------------------------------------

ideal_best = np.array([

    weighted["average_rating"].max(),

    weighted["final_score"].max(),

    weighted["distance_km"].min(),

    weighted["average_price_pkr"].min(),

    weighted["avg_delivery_time"].min()

])

# ---------------------------------------------
# Ideal Worst
# ---------------------------------------------

ideal_worst = np.array([

    weighted["average_rating"].min(),

    weighted["final_score"].min(),

    weighted["distance_km"].max(),

    weighted["average_price_pkr"].max(),

    weighted["avg_delivery_time"].max()

])

# ---------------------------------------------
# Euclidean Distance
# ---------------------------------------------

distance_best = np.sqrt(

    ((weighted - ideal_best) ** 2).sum(axis=1)

)

distance_worst = np.sqrt(

    ((weighted - ideal_worst) ** 2).sum(axis=1)

)

# ---------------------------------------------
# Relative Closeness
# ---------------------------------------------

restaurant_scores["topsis_score"] = (

    distance_worst

    /

    (distance_best + distance_worst)

)

# ---------------------------------------------
# Convert to %
# ---------------------------------------------

restaurant_scores["match_score"] = (

    restaurant_scores["topsis_score"]

    * 100

).round(1)

# ---------------------------------------------
# Final Ranking
# ---------------------------------------------

restaurant_scores = restaurant_scores.sort_values(

    "topsis_score",

    ascending=False

).reset_index(drop=True)


def recommendation_reason(row):

    reasons = []

    if row["distance_km"] <= 50:
        reasons.append("Near your location")

    if row["average_rating"] >= 4.5:
        reasons.append("Highly rated")

    if row["avg_delivery_time"] <= 45:
        reasons.append("Fast delivery")

    if row["average_price_pkr"] <= 1000:
        reasons.append("Affordable")

    return " • ".join(reasons)


restaurant_scores["reason"] = restaurant_scores.apply(

    recommendation_reason,

    axis=1

)

# # =====================================================
# # DISPLAY
# # =====================================================

# st.subheader(
#     "🏆 Top Restaurants"
# )

# st.dataframe(
#     restaurant_scores[
#         [
#             "restaurant_name",
#             "average_rating",
#             "average_price_pkr",
#             "avg_delivery_time",
#             "offers_delivery",
#             "topsis_score"
#         ]
#     ],
#     use_container_width=True
# )


# =====================================================
# PROFESSIONAL RESTAURANT CARDS
# =====================================================

st.markdown("---")
st.subheader("🏆 Top Nearby Restaurant Recommendations")

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
    restaurant_scores.head(10).iterrows(),
    start=1
):

    medal = medals[rank-1]

    delivery = (
        "Available ✅"
        if row["offers_delivery"] == 1
        else
        "Not Available ❌"
    )

    # -----------------------------------
    # Explainability
    # -----------------------------------

    reasons = []

    if row["distance_km"] <= 50:
        reasons.append("📍 Near your location")

    if row["average_rating"] >= 4.5:
        reasons.append("⭐ Highly rated")

    if row["avg_delivery_time"] <= 45:
        reasons.append("🚚 Fast delivery")

    if row["average_price_pkr"] <= 1000:
        reasons.append("💰 Budget friendly")

    reasons.append("🍽 Matches your recommended food")

    explanation = "<br>".join(reasons)

    # -----------------------------------
    # Card
    # -----------------------------------

    st.markdown(
        f"""
<div style="
background-color:#FFOOFF;
padding:22px;
border-radius:15px;
margin-bottom:20px;
box-shadow:0px 5px 18px rgba(0,0,0,0.12);
border-left:8px solid #ff6b35;
">

<h2>{medal} {row['restaurant_name']}</h2>

<hr>

<b>🍔 Recommended Food</b><br>
{row['food_name']}

<br><br>

<b>📍 Distance</b><br>
{row['distance_km']:.2f} KM away

<br><br>

<b>🚗 Estimated Drive Time</b><br>
{row['travel_time']} minutes

<br><br>

<b>⭐ Restaurant Rating</b><br>
{row['average_rating']} / 5

<br><br>

<b>💰 Average Price</b><br>
PKR {row['average_price_pkr']:.0f}

<br><br>

<b>🚚 Delivery</b><br>
{delivery}

<br><br>

<b>⏱ Average Delivery Time</b><br>
{row['avg_delivery_time']} minutes

<br><br>

<b>🏆 Match Score</b><br>

<span style="
font-size:26px;
font-weight:bold;
color:#00FFFF;
">
{row['match_score']}%
</span>

<br><br>

<b>💡 Why Recommended?</b>

<br>

{explanation}

</div>
""",
        unsafe_allow_html=True
    )

    # -----------------------------------
    # Save Restaurant ID
    # -----------------------------------

    execute_query(
        """
        UPDATE Recommendation_Items
        SET restaurant_id=%s
        WHERE history_id=%s
        AND food_id=%s
        """,
        (
            int(row["restaurant_id"]),
            history_id,
            int(row["food_id"])
        )
    )

# =====================================================
# INTERACTIVE RESTAURANT MAP
# =====================================================

st.markdown("---")

st.subheader("🗺 Nearby Restaurants Map")

# -----------------------------------------
# Create Map
# -----------------------------------------

restaurant_map = folium.Map(

    location=[
        user_lat,
        user_lon
    ],

    zoom_start=14,

    control_scale=True

)

# -----------------------------------------
# USER LOCATION
# -----------------------------------------

folium.Marker(

    location=[
        user_lat,
        user_lon
    ],

    popup="""
<b>You are here</b>
""",

    tooltip="Current Location",

    icon=folium.Icon(

        color="blue",

        icon="user",

        prefix="fa"

    )

).add_to(

    restaurant_map

)


# -----------------------------------------
# RESTAURANT MARKERS (Ranked)
# -----------------------------------------

for rank, (_, row) in enumerate(
    restaurant_scores.head(10).iterrows(),
    start=1
):

    # -------------------------------------
    # Marker Color
    # -------------------------------------

    if rank == 1:

        marker_color = "orange"      # Gold

        medal = "🥇"

    elif rank == 2:

        marker_color = "lightgray"   # Silver

        medal = "🥈"

    elif rank == 3:

        marker_color = "lightred"       # Bronze

        medal = "🥉"

    else:

        marker_color = "red"

        medal = f"#{rank}"

    popup_html = f"""
    <h4>{medal} {row['restaurant_name']}</h4>

    <hr>

    <b>🍔 Recommended Food</b><br>
    {row['food_name']}

    <br><br>

    <b>📍 Distance</b><br>
    {row['distance_km']:.2f} KM

    <br><br>

    <b>🚗 Estimated Drive Time</b><br>
    {row['travel_time']} mins

    <br><br>

    <b>⭐ Rating</b><br>
    {row['average_rating']} / 5

    <br><br>

    <b>💰 Average Price</b><br>
    PKR {row['average_price_pkr']:.0f}

    <br><br>

    <b>🏆 Match Score</b><br>

    <span style="font-size:20px;
                 color:green;
                 font-weight:bold;">
        {row['match_score']}%
    </span>
    """

    folium.Marker(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        popup=folium.Popup(
            popup_html,
            max_width=320
        ),

        tooltip=f"{medal} {row['restaurant_name']}",

        icon=folium.Icon(
            color=marker_color,
            icon="cutlery",
            prefix="fa"
        )

    ).add_to(restaurant_map)


# -----------------------------------------
# DISTANCE LINES
# -----------------------------------------

for _, row in restaurant_scores.head(10).iterrows():

    distance = row["distance_km"]

    if distance <= 3:

        line_color = "green"

    elif distance <= 7:

        line_color = "orange"

    else:

        line_color = "red"

    folium.PolyLine(

        locations=[
            [user_lat, user_lon],
            [row["latitude"], row["longitude"]]
        ],

        color=line_color,

        weight=3,

        opacity=0.8

    ).add_to(restaurant_map)
# -----------------------------------------
# SHOW MAP
# -----------------------------------------

st.markdown("""

### 🗺 Map Legend

🥇 **Gold Marker** → Best Restaurant

🥈 **Silver Marker** → Second Best

🥉 **Bronze Marker** → Third Best

🔴 **Red Marker** → Other Recommended Restaurants

🔵 **Blue Marker** → Your Current Location

🟢 **Green Line** → Less than 3 KM

🟠 **Orange Line** → 3–7 KM

🔴 **Red Line** → More than 7 KM

""")

st_folium(

    restaurant_map,

    width=1200,

    height=650

)


# =====================================================
# CONTINUE
# =====================================================

if st.button(
    "➡ Continue To Feedback"
):

    st.switch_page(
        "pages/6_Feedback.py"
    )