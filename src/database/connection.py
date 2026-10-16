import logging

import psycopg2

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def get_connection():
    """Create a PostgreSQL database connection."""
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10,
        )
    except psycopg2.Error as error:
        logger.error(
            "Database connection failed: %s",
            type(error).__name__,
        )
        raise DatabaseError("Database connection failed.") from error


def execute_query(query, params=None, fetch=False):
    """Execute a parameterized query and manage its transaction."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

            if fetch:
                return cursor.fetchall()

            connection.commit()
            logger.info("Database query executed successfully")
            return None

    except psycopg2.Error as error:
        logger.error(
            "Database operation failed: %s",
            type(error).__name__,
        )
        connection.rollback()
        raise DatabaseError("Database operation failed.") from error

    finally:
        connection.close()