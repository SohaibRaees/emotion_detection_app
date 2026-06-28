# =====================================================
# DATABASE CONNECTION MODULE
# Personalized Emotion-Aware Food Recommendation System
# =====================================================

import mysql.connector
import pandas as pd


# =====================================================
# CREATE DATABASE CONNECTION
# =====================================================

def get_connection():
    """
    Returns a MySQL database connection.
    """

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="sohaib12",
        database="emotion_food_recommendation"
    )

    return connection


# =====================================================
# FETCH DATA (SELECT QUERIES)
# =====================================================

def fetch_data(query):
    """
    Executes SELECT query and returns DataFrame.
    """

    conn = get_connection()

    df = pd.read_sql(
        query,
        conn
    )

    conn.close()

    return df


# =====================================================
# EXECUTE INSERT / UPDATE / DELETE
# =====================================================

def execute_query(query, values=None):
    """
    Executes INSERT, UPDATE, DELETE queries.
    """

    conn = get_connection()

    cursor = conn.cursor()

    if values:
        cursor.execute(query, values)
    else:
        cursor.execute(query)

    conn.commit()

    cursor.close()
    conn.close()


# =====================================================
# INSERT AND RETURN LAST INSERT ID
# =====================================================

def execute_insert(query, values):
    """
    Executes INSERT query and returns inserted ID.
    """

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(query, values)

    conn.commit()

    last_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return last_id


# =====================================================
# TEST CONNECTION
# =====================================================

if __name__ == "__main__":

    try:

        conn = get_connection()

        if conn.is_connected():

            print(
                "✅ Connected to MySQL Database Successfully"
            )

            conn.close()

    except Exception as e:

        print(
            f"❌ Database Connection Error: {e}"
        )