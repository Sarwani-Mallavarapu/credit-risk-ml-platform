import os
import logging
import psycopg2

logger = logging.getLogger(__name__)

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD
)

class DatabaseError(Exception):
    """Raised when a database operation fails."""

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
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            logger.info("Database query executed successfully")
            if fetch:
                return cursor.fetchall()

            connection.commit()
            return None

    except psycopg2.Error as error:
        logger.error("Database operation failed", exc_info=True)
        connection.rollback()
        raise  DatabaseError("Database operation failed.") from error

    finally:
        connection.close()