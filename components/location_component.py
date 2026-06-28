# =====================================================
# LOCATION COMPONENT
# =====================================================

import streamlit as st

from streamlit_js_eval import streamlit_js_eval


# =====================================================
# GET CURRENT LOCATION
# =====================================================

def get_browser_location():
    """
    Detect current browser GPS location.

    Returns:
        latitude,
        longitude

    If permission denied:
        returns (None, None)
    """

    location = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve, reject) => {

            if (!navigator.geolocation) {

                resolve(null);

            } else {

                navigator.geolocation.getCurrentPosition(

                    (position) => {

                        resolve({

                            latitude: position.coords.latitude,

                            longitude: position.coords.longitude

                        });

                    },

                    (error) => {

                        resolve(null);

                    }

                );

            }

        })
        """,
        key="browser_location"
    )

    if location is None:

        return None, None

    if not isinstance(location, dict):

        return None, None

    latitude = location.get("latitude")
    longitude = location.get("longitude")

    return latitude, longitude


# =====================================================
# DISPLAY LOCATION
# =====================================================

def show_current_location():

    lat, lon = get_browser_location()

    if lat is None:

        st.warning(
            "Location permission denied or unavailable."
        )

        return None, None

    st.success("📍 Current Location")

    st.write(f"Latitude : {lat:.6f}")
    st.write(f"Longitude: {lon:.6f}")

    return lat, lon