import pandas as pd

from src.database.connection import execute_query
from src.features.engineering import engineer_features

import logging
logger = logging.getLogger(__name__)

import time


def get_risk_band(default_probability: float) -> str:
    if default_probability < 0.30:
        return "LOW"
    elif default_probability < 0.55:
        return "MEDIUM"
    else:
        return "HIGH"


def generate_prediction(
    applicant_data: dict,
    model,
    model_config: dict
) -> dict:

    start_time = time.perf_counter()
    input_df = pd.DataFrame([applicant_data])

    input_df = engineer_features(input_df)

    default_probability = model.predict_proba(
        input_df
    )[:, 1][0]

    candidate_threshold = model_config["candidate_threshold"]

    predicted_default = (
        default_probability >= candidate_threshold
    )

    risk_band = get_risk_band(default_probability)

    save_query = """
        INSERT INTO predictions (
            applicant_id,
            default_probability,
            predicted_default,
            risk_band,
            model_version
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    execute_query(
        save_query,
        (
            applicant_data["applicant_id"],
            float(default_probability),
            bool(predicted_default),
            risk_band,
            model_config["model_version"]
        )
    )

    logger.info(
        "Prediction generated for applicant=%s model_version=%s "
        "risk_band=%s probability=%.4f",
        applicant_data["applicant_id"],
        model_config["model_version"],
        risk_band,
        float(default_probability),
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    return {
        "applicant_id": applicant_data["applicant_id"],
        "default_probability": round(
            float(default_probability),
            4
        ),
        "predicted_default": bool(predicted_default),
        "risk_band": risk_band,
        "model_version": model_config["model_version"],
        "prediction_duration_ms": duration_ms
    }