import os
import psycopg2


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "credit_risk"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )


def execute_query(query, params=None, fetch=False):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

            if fetch:
                return cursor.fetchall()

            connection.commit()
            return None

    finally:
        connection.close()