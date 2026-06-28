# =====================================================
# LOCATION SERVICE
# =====================================================
import requests
import math
import streamlit as st

from components.location_component import (
    get_browser_location
)

from database.db_connection import (
    fetch_data,
    execute_query
)

# =====================================================
# UPDATE USER LOCATION
# =====================================================

def update_user_location(user_id):
    """
    Detect browser location and update Users table.
    """

    latitude, longitude = get_browser_location()

    if latitude is None or longitude is None:
        return False

    execute_query(
        """
        UPDATE Users
        SET
            latitude=%s,
            longitude=%s
        WHERE user_id=%s
        """,
        (
            latitude,
            longitude,
            user_id
        )
    )

    st.session_state.latitude = latitude
    st.session_state.longitude = longitude

    return True


# =====================================================
# LOAD USER LOCATION
# =====================================================

def get_user_location(user_id):
    """
    Returns latitude and longitude from database.
    """

    df = fetch_data(
        f"""
        SELECT
            latitude,
            longitude
        FROM Users
        WHERE user_id={user_id}
        """
    )

    if df.empty:
        return None, None

    latitude = df.iloc[0]["latitude"]
    longitude = df.iloc[0]["longitude"]

    if latitude is None or longitude is None:
        return None, None

    return float(latitude), float(longitude)


# =====================================================
# HAVERSINE DISTANCE
# =====================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two GPS coordinates.
    Returns distance in KM.
    """

    R = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# =====================================================
# ADD DISTANCE COLUMN
# =====================================================

def add_distance_to_restaurants(
    restaurants_df,
    user_lat,
    user_lon
):
    """
    Adds a distance_km column to the Restaurants dataframe.
    """

    restaurants = restaurants_df.copy()

    restaurants["distance_km"] = restaurants.apply(

        lambda row: haversine_distance(

            user_lat,

            user_lon,

            row["latitude"],

            row["longitude"]

        ),

        axis=1

    )

    return restaurants


# =====================================================
# FILTER NEARBY RESTAURANTS
# =====================================================

def filter_nearby_restaurants(
    restaurants_df,
    max_distance_km=15
):
    """
    Returns only nearby restaurants.
    """

    return restaurants_df[
        restaurants_df["distance_km"] <= max_distance_km
    ].copy()


# =====================================================
# SHOW CURRENT LOCATION
# =====================================================

def display_user_location(user_id):

    lat, lon = get_user_location(user_id)

    if lat is None:

        st.warning(
            "Current location unavailable."
        )

        return

    st.success("📍 Current Location")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Latitude",
            f"{lat:.6f}"
        )

    with col2:
        st.metric(
            "Longitude",
            f"{lon:.6f}"
        )

# =====================================================
# GET CITY NAME
# =====================================================

def get_city_name(latitude, longitude):
    """
    Reverse geocoding using OpenStreetMap.
    Returns city/state/country.
    """

    try:

        url = (
            "https://nominatim.openstreetmap.org/reverse"
        )

        params = {

            "format": "json",

            "lat": latitude,

            "lon": longitude

        }

        headers = {

            "User-Agent":
            "EmotionFoodRecommendationApp"

        }

        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=10

        )

        if response.status_code != 200:

            return "Unknown"

        data = response.json()

        address = data.get("address", {})

        city = (

            address.get("city")

            or address.get("town")

            or address.get("village")

            or address.get("municipality")

            or address.get("county")

        )

        state = address.get("state")

        country = address.get("country")

        if city:

            return f"{city}, {country}"

        if state:

            return f"{state}, {country}"

        return country

    except:

        return "Unknown"