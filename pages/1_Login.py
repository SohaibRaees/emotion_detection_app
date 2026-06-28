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
# LOGIN PAGE
# =====================================================

import streamlit as st
import bcrypt

from database.db_connection import fetch_data

st.page_link(
    "pages/2_Registration.py",
    label="Create New Account"
)

st.title("🔐 User Login")

email = st.text_input("Email")

password = st.text_input(
    "Password",
    type="password"
)


if st.button("Login"):

    email = email.strip().lower()

    user = fetch_data(
        f"""
        SELECT *
        FROM Users
        WHERE LOWER(email)='{email}'
        """
    )

    if len(user) == 0:

        st.error("User not found.")

    else:

        stored_hash = user.iloc[0]["password_hash"]

        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        ):

            st.session_state.user_id = int(
                user.iloc[0]["user_id"]
            )

            st.session_state.user_name = str(
                user.iloc[0]["full_name"]
            )

            st.success(
                f"Welcome {st.session_state.user_name}"
            )

        else:

            st.error(
                "Invalid Password"
            )

        st.switch_page("pages/3_Emotion_Detection.py")