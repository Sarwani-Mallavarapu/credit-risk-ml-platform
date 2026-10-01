import os

import psycopg2

class DatabaseError(Exception):
    """Raised when a database operation fails."""

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

    except psycopg2.Error as error:
        connection.rollback()
        raise  DatabaseError("Database operation failed.") from error

    finally:
        connection.close()