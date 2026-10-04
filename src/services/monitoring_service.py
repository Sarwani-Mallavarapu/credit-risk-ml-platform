from src.database.connection import execute_query

import pandas as pd

from src.database.connection import execute_query


FEATURE_MAPPING = {
    "X1": "credit_limit",
    "X2": "gender",
    "X3": "education",
    "X4": "marital_status",
    "X5": "age",

    "X6": "repay_sep",
    "X7": "repay_aug",
    "X8": "repay_jul",
    "X9": "repay_jun",
    "X10": "repay_may",
    "X11": "repay_apr",

    "X12": "bill_sep",
    "X13": "bill_aug",
    "X14": "bill_jul",
    "X15": "bill_jun",
    "X16": "bill_may",
    "X17": "bill_apr",

    "X18": "payment_sep",
    "X19": "payment_aug",
    "X20": "payment_jul",
    "X21": "payment_jun",
    "X22": "payment_may",
    "X23": "payment_apr",
}

def calculate_psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = reference.quantile(
        [i / bins for i in range(bins + 1)]
    ).unique()

    if len(breakpoints) < 2:
        return 0.0

    reference_bins = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    )

    current_bins = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    )

    reference_distribution = (
        reference_bins.value_counts(normalize=True)
        .sort_index()
    )

    current_distribution = (
        current_bins.value_counts(normalize=True)
        .sort_index()
    )

    reference_distribution = reference_distribution.replace(0, 0.0001)
    current_distribution = current_distribution.replace(0, 0.0001)

    psi = (
        (
            current_distribution
            - reference_distribution
        )
        * (
            current_distribution
            / reference_distribution
        ).apply(lambda x: __import__("math").log(x))
    ).sum()

    return float(psi)

def get_feature_drift(days: int = 7) -> dict:
    reference_path = (
        "data/processed/credit_card_default_cleaned.csv"
    )

    reference_df = pd.read_csv(reference_path)

    query = """
        SELECT input_features
        FROM predictions
        WHERE created_at >= CURRENT_TIMESTAMP
            - (%s * INTERVAL '1 day')
    """

    rows = execute_query(
        query,
        (days,),
        fetch=True
    )

    current_records = [
        row[0]
        for row in rows
        if row[0] is not None
    ]

    sample_size = len(current_records)

    if sample_size < 30:
        return {
            "sample_size": sample_size,
            "status": "INSUFFICIENT_DATA",
            "minimum_required": 30,
            "features": []
        }

    current_df = pd.DataFrame(current_records)

    drift_results = []

    for reference_feature, api_feature in FEATURE_MAPPING.items():

        if api_feature not in current_df.columns:
            continue

        psi = calculate_psi(
            reference_df[reference_feature],
            current_df[api_feature]
        )

        if psi < 0.10:
            drift_level = "LOW"
        elif psi < 0.25:
            drift_level = "MEDIUM"
        else:
            drift_level = "HIGH"

        drift_results.append({
            "feature": api_feature,
            "reference_feature": reference_feature,
            "psi": round(psi, 4),
            "drift_level": drift_level
        })

    return {
        "sample_size": sample_size,
        "status": "OK",
        "minimum_required": 30,
        "features": drift_results
    }

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