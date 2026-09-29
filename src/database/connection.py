import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def execute_query(query, params=None, fetch=False):
    connection = get_connection()

    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)

            if fetch:
                result = cursor.fetchall()
            else:
                result = None

            connection.commit()
            return result

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()