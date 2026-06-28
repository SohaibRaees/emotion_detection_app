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


import streamlit as st

from components.location_component import (
    show_current_location
)

st.title("Location Test")

show_current_location()