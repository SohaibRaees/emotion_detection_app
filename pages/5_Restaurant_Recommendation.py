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

from utils.ui import (
    load_ui,
    page_header,
    page_intro
)

load_ui("Restaurant Recommendations")

page_intro(

    "Restaurant Recommendation",

    "Discover nearby restaurants ofAfering your recommended meals.",

    "🍴"

)

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

recommended_foods = fetch_data(f"""
SELECT

ri.food_id,

ri.final_score,

ri.rank_position,

f.food_name

FROM Recommendation_Items ri

JOIN Foods f

ON ri.food_id=f.food_id

WHERE history_id={history_id}

ORDER BY rank_position
""")

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

candidate_restaurants = add_distance_to_restaurants(

    candidate_restaurants,

    user_lat,

    user_lon

)

AVERAGE_CITY_SPEED = 30

candidate_restaurants["travel_time"]=(

candidate_restaurants["distance_km"]

/

AVERAGE_CITY_SPEED

)*60

candidate_restaurants["travel_time"]=(

candidate_restaurants["travel_time"]

.round()

.astype(int)

)

closest_distance = candidate_restaurants["distance_km"].min()

closest_name = candidate_restaurants.loc[
    candidate_restaurants["distance_km"].idxmin(),
    "restaurant_name"
]

city = get_city_name(user_lat, user_lon)

st.success("📍 Current Location")
st.info(f"🏙 City: {city}")

col1,col2,col3,col4=st.columns(4)

with col1:

    st.metric(

        "📍 Current City",

        city

    )

with col2:

    st.metric(

        "🍴 Restaurants",

        len(candidate_restaurants)

    )

with col3:

    st.metric(

        "📍 Closest",

        f"{closest_distance:.1f} KM"

    )

with col4:

    st.metric(

        "⭐ Best Rating",

        f"{candidate_restaurants['average_rating'].max():.1f}"

    )

st.markdown("---")

# =====================================================
# FILTER BY DISTANCE
# =====================================================

with st.expander(

    "⚙ Recommendation Settings",

    expanded=False

):

    MAX_DISTANCE=st.slider(

        "Maximum Distance (KM)",

        1,

        200,

        50

    )

    min_rating=st.slider(

        "Minimum Restaurant Rating",

        1.0,

        5.0,

        3.5,

        step=0.1

    )

    delivery_only=st.checkbox(

        "Delivery Only",

        False

    )

# -------

candidate_restaurants=filter_nearby_restaurants(
    candidate_restaurants,MAX_DISTANCE)

candidate_restaurants=candidate_restaurants[
    candidate_restaurants["average_rating"] >= min_rating]

if delivery_only:
    candidate_restaurants=candidate_restaurants[
    candidate_restaurants["offers_delivery"]==1]

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
restaurant_scores=(

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

"food_name":"first",

"food_id":"first",

"latitude":"first",

"longitude":"first",

"final_score":"mean"

}

)

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



#  Restaurant Recommendation Statistics

col1,col2,col3,col4=st.columns(4)

with col1:

    st.metric(

        "🏆 Best Match",

        f"{restaurant_scores.iloc[0]['match_score']:.1f}%"

    )

with col2:

    st.metric(

        "⭐ Highest Rating",

        restaurant_scores["average_rating"].max()

    )

with col3:

    st.metric(

        "📍 Closest",

        f"{restaurant_scores['distance_km'].min():.1f} KM"

    )

with col4:

    st.metric(

        "🚗 Fastest",

        f"{restaurant_scores['travel_time'].min()} mins"

    )

st.markdown("---")

# =====================================================
# PREMIUM RESTAURANT CARDS
# =====================================================

st.markdown("---")

st.subheader("🍴 Best Restaurants For You")

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
        "🚚 Delivery Available"
        if row["offers_delivery"] == 1
        else
        "🏪 Pickup Only"
    )

    badges = []

    if row["distance_km"] <= 3:
        badges.append("📍 Nearby")

    elif row["distance_km"] <= 7:
        badges.append("🚗 Short Drive")

    else:
        badges.append("🛣 Long Distance")

    if row["average_rating"] >= 4.7:
        badges.append("⭐ Highly Rated")

    elif row["average_rating"] >= 4.3:
        badges.append("👍 Popular")

    if row["avg_delivery_time"] <= 25:
        badges.append("⚡ Fast Delivery")

    if row["average_price_pkr"] <= 900:
        badges.append("💰 Affordable")

    badges.append("❤️ Emotion Match")

    badge_html = "".join(
        [
            f"<span class='badge'>{badge}</span>"
            for badge in badges
        ]
    )

    progress = min(
        row["match_score"],
        100
    )

    st.markdown(
        f"""
<div class="restaurant-card">

<div style="display:flex;justify-content:space-between;align-items:center;">

<div>

<div class="restaurant-title">

{medal} {row['restaurant_name']}

</div>

<div class="restaurant-food">

🍔 Recommended Food:
<b>{row['food_name']}</b>

</div>

</div>

<div class="match-score">

{row['match_score']:.1f}%

</div>

</div>

<br>

<div style="
display:flex;
justify-content:space-between;
flex-wrap:wrap;
font-size:15px;
">

<div>

📍 <b>{row['distance_km']:.1f} km</b>

</div>

<div>

🚗 <b>{row['travel_time']} min</b>

</div>

<div>

⭐ <b>{row['average_rating']}</b>

</div>

<div>

💰 <b>PKR {row['average_price_pkr']:.0f}</b>

</div>

<div>

🚚 <b>{row['avg_delivery_time']} min</b>

</div>

</div>

<br>

<div class="progress">

<div

class="progress-fill"

style="width:{progress}%">

</div>

</div>

<br>

{badge_html}

<br><br>

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div>

<b>Recommendation Reason</b>

<br>

<small>

{row['reason']}

</small>

</div>

<div style="
font-size:15px;
font-weight:bold;
color:#16A34A;
">

🏆 TOPSIS Score

<br>

{row['topsis_score']:.4f}

</div>

</div>

</div>

""",
        unsafe_allow_html=True
    )

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
# INTERACTIVE MAP SECTION
# =====================================================

st.markdown("---")

st.markdown("""
<div class="section-header">

🗺 Nearby Restaurant Map

</div>

<div style="color:#64748B;
margin-bottom:15px;">

Explore recommended restaurants around your current location.
Click any marker to view detailed information.

</div>
""",
unsafe_allow_html=True)

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

    popup_html=f"""

        <div style="width:250px">

        <h3>

        {medal}

        {row['restaurant_name']}

        </h3>

        <hr>

        <b>🍔 Recommended Food</b>

        <br>

        {row['food_name']}

        <br><br>

        <b>📍 Distance</b>

        <br>

        {row['distance_km']:.1f} KM

        <br><br>

        <b>🚗 Travel Time</b>

        <br>

        {row['travel_time']} Minutes

        <br><br>

        <b>⭐ Rating</b>

        <br>

        {row['average_rating']}

        <br><br>

        <b>💰 Average Price</b>

        <br>

        PKR {row['average_price_pkr']:.0f}

        <br><br>

        <b>🏆 Match Score</b>

        <br>

        <span style="font-size:22px;
        font-weight:bold;
        color:#16A34A">

        {row['match_score']}%

        </span>

        </div>

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

<div class="custom-card">

<h3>🧭 Map Legend</h3>

| Icon | Meaning |
|------|---------|
| 🔵 | Your Current Location |
| 🥇 | Best Restaurant |
| 🥈 | Second Recommendation |
| 🥉 | Third Recommendation |
| 🔴 | Other Recommended Restaurants |
| 🟢 Line | Less than 3 KM |
| 🟠 Line | Between 3–7 KM |
| 🔴 Line | More than 7 KM |

</div>

""",
unsafe_allow_html=True)


st_folium(

restaurant_map,

width=None,

height=700,

returned_objects=[]

)



st.markdown("---")

st.subheader("📊 Recommendation Statistics")

c1,c2,c3,c4=st.columns(4)

with c1:

    st.metric(

        "Restaurants",

        len(restaurant_scores)

    )

with c2:

    st.metric(

        "Average Rating",

        f"{restaurant_scores['average_rating'].mean():.2f}"

    )

with c3:

    st.metric(

        "Average Distance",

        f"{restaurant_scores['distance_km'].mean():.1f} KM"

    )

with c4:

    st.metric(

        "Average Match",

        f"{restaurant_scores['match_score'].mean():.1f}%"

    )
# =====================================================
# CONTINUE
# =====================================================
st.markdown("---")

st.markdown("""

<div class="custom-card">
<h2 style="text-align:center">
🎉 RECOMMENDATION COMPLETED
</h2>
<p style="text-align:center">
Please continue to provide your valuable feedback.
</p>
</div>

""",
unsafe_allow_html=True)

col1,col2,col3=st.columns([1,2,1])

with col2:
    if st.button(
        "⭐ Continue To Feedback",
        use_container_width=True
    ):
        st.switch_page(
            "pages/6_Feedback.py"
        )