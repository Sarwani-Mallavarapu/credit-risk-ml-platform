from src.database.connection import execute_query


def get_prediction_summary() -> dict:
    query = """
        SELECT
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (
                WHERE predicted_default = TRUE
            ) AS predicted_defaults,
            AVG(default_probability) AS average_probability,
            COUNT(*) FILTER (
                WHERE risk_band = 'LOW'
            ) AS low_risk,
            COUNT(*) FILTER (
                WHERE risk_band = 'MEDIUM'
            ) AS medium_risk,
            COUNT(*) FILTER (
                WHERE risk_band = 'HIGH'
            ) AS high_risk
        FROM predictions
    """

    result = execute_query(
        query,
        fetch=True
    )[0]

    total_predictions = result[0] or 0
    predicted_defaults = result[1] or 0
    average_probability = result[2] or 0

    predicted_default_rate = (
        predicted_defaults / total_predictions
        if total_predictions > 0
        else 0
    )

    return {
        "total_predictions": total_predictions,
        "predicted_defaults": predicted_defaults,
        "predicted_default_rate": round(
            predicted_default_rate,
            4
        ),
        "average_probability": round(
            float(average_probability),
            4
        ),
        "risk_distribution": {
            "LOW": result[3] or 0,
            "MEDIUM": result[4] or 0,
            "HIGH": result[5] or 0
        }
    }

def get_prediction_trend() -> dict:
    query = """
        SELECT
            COUNT(*) FILTER (
                WHERE created_at >= CURRENT_DATE
            ) AS today,

            COUNT(*) FILTER (
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
            ) AS last_7_days,

            COUNT(*) FILTER (
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            ) AS last_30_days
        FROM predictions
    """

    result = execute_query(
        query,
        fetch=True
    )[0]

    return {
        "today": result[0] or 0,
        "last_7_days": result[1] or 0,
        "last_30_days": result[2] or 0
    }

def get_prediction_summary_by_period(days: int) -> dict:
    query = """
        SELECT
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (
                WHERE predicted_default = TRUE
            ) AS predicted_defaults,
            AVG(default_probability) AS average_probability,
            COUNT(*) FILTER (
                WHERE risk_band = 'LOW'
            ) AS low_risk,
            COUNT(*) FILTER (
                WHERE risk_band = 'MEDIUM'
            ) AS medium_risk,
            COUNT(*) FILTER (
                WHERE risk_band = 'HIGH'
            ) AS high_risk
        FROM predictions
        WHERE created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
    """

    result = execute_query(
        query,
        (days,),
        fetch=True
    )[0]

    total_predictions = result[0] or 0
    predicted_defaults = result[1] or 0
    average_probability = result[2] or 0

    predicted_default_rate = (
        predicted_defaults / total_predictions
        if total_predictions > 0
        else 0
    )

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "predicted_defaults": predicted_defaults,
        "predicted_default_rate": round(
            predicted_default_rate,
            4
        ),
        "average_probability": round(
            float(average_probability),
            4
        ),
        "risk_distribution": {
            "LOW": result[3] or 0,
            "MEDIUM": result[4] or 0,
            "HIGH": result[5] or 0
        }
    }

def get_prediction_trend(days: int = 7) -> list:
    query = """
        SELECT
            DATE(created_at) AS prediction_date,
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (
                WHERE predicted_default = TRUE
            ) AS predicted_defaults,
            AVG(default_probability) AS average_probability
        FROM predictions
        WHERE created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
        GROUP BY DATE(created_at)
        ORDER BY prediction_date
    """

    rows = execute_query(
        query,
        (days,),
        fetch=True
    )

    trend = []

    for row in rows:
        total_predictions = row[1] or 0
        predicted_defaults = row[2] or 0

        predicted_default_rate = (
            predicted_defaults / total_predictions
            if total_predictions > 0
            else 0
        )

        trend.append({
            "date": row[0].isoformat(),
            "total_predictions": total_predictions,
            "predicted_defaults": predicted_defaults,
            "predicted_default_rate": round(
                predicted_default_rate,
                4
            ),
            "average_probability": round(
                float(row[3] or 0),
                4
            )
        })

    return trend