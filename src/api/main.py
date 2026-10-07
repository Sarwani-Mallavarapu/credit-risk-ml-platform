from fastapi.responses import JSONResponse
import joblib
import pandas as pd

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.engineering import engineer_features
from src.config import CONFIG_PATH, MODEL_PATH
from src.database.connection import execute_query, DatabaseError
from src.services.prediction_service import generate_prediction
# from src.services.monitoring_service import get_prediction_summary


from src.services.monitoring_service import (
    get_feature_drift,
    get_prediction_summary,
    get_prediction_trend,
    get_prediction_summary_by_period,
)

from src.services.performance_monitoring_service import (
    get_model_performance,
)

import logging
import time

from src.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Load model and configuration
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
model_config = joblib.load(CONFIG_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit default risk.",
    version=model_config["model_version"]
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------
class ApplicantRequest(BaseModel):
    credit_limit: float = Field(ge=0)
    gender: int = Field(ge=1, le=2)
    education: int = Field(ge=0, le=6)
    marital_status: int = Field(ge=0, le=3)
    age: int = Field(ge=18, le=100)
    applicant_id: str = Field(min_length=1)

    repay_sep: int = Field(ge=-2, le=8)
    repay_aug: int = Field(ge=-2, le=8)
    repay_jul: int = Field(ge=-2, le=8)
    repay_jun: int = Field(ge=-2, le=8)
    repay_may: int = Field(ge=-2, le=8)
    repay_apr: int = Field(ge=-2, le=8)

    bill_sep: float = Field(ge=0)
    bill_aug: float = Field(ge=0)
    bill_jul: float = Field(ge=0)
    bill_jun: float = Field(ge=0)
    bill_may: float = Field(ge=0)
    bill_apr: float = Field(ge=0)

    payment_sep: float = Field(ge=0)
    payment_aug: float = Field(ge=0)
    payment_jul: float = Field(ge=0)
    payment_jun: float = Field(ge=0)
    payment_may: float = Field(ge=0)
    payment_apr: float = Field(ge=0)

class BatchApplicantRequest(BaseModel):
    applicants: list[ApplicantRequest]

class OutcomeRequest(BaseModel):
    actual_default: bool

# --------------------------------------------------
# Risk-band logic
# --------------------------------------------------

def get_risk_band(default_probability: float) -> str:
    if default_probability < 0.30:
        return "LOW"
    elif default_probability < 0.55:
        return "MEDIUM"
    else:
        return "HIGH"


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": model_config["model_version"]
    }


# --------------------------------------------------
# Model information endpoint
# --------------------------------------------------

@app.get("/model/info")
def model_info():
    return {
        "model_version": model_config["model_version"],
        "model_type": model_config["model_type"],
        "candidate_threshold": model_config["candidate_threshold"],
        "risk_band_boundaries": model_config["risk_band_boundaries"]
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(applicant: ApplicantRequest):

    applicant_data = applicant.model_dump()

    return generate_prediction(
        applicant_data,
        model,
        model_config
    )


@app.get("/predictions/{applicant_id}")
def get_prediction(applicant_id: str):
    query = """
        SELECT
            id,
            applicant_id,
            default_probability,
            predicted_default,
            risk_band,
            model_version,
            created_at
        FROM predictions
        WHERE applicant_id = %s
        ORDER BY created_at DESC
    """

    results = execute_query(
        query,
        (applicant_id,),
        fetch=True
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for applicant: {applicant_id}"
        )

    return {
        "applicant_id": applicant_id,
        "predictions": results
    }


@app.post("/predict/batch")
def predict_batch(batch: BatchApplicantRequest):

    results = []

    for applicant in batch.applicants:
        applicant_data = applicant.model_dump()

        prediction = generate_prediction(
            applicant_data,
            model,
            model_config
        )

        results.append(prediction)

    return {
        "count": len(results),
        "predictions": results
    }

@app.exception_handler(DatabaseError)
def database_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database operation failed."
        },
    )

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "HTTP request method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response

@app.get("/monitoring/summary")
def monitoring_summary():
    return get_prediction_summary()

@app.get("/monitoring/summary/period")
def monitoring_summary_period(days: int = 7):
    return get_prediction_summary_by_period(days)

@app.get("/monitoring/trend")
def monitoring_trend(days: int = 7):
    return get_prediction_trend(days)

@app.get("/monitoring/drift")
def monitoring_drift(days: int = 7):
    return get_feature_drift(days)


@app.get("/predictions/{applicant_id}")
@app.patch("/predictions/{applicant_id}/outcome")
def update_prediction_outcome(
    applicant_id: str,
    outcome: OutcomeRequest
):
    query = """
        UPDATE predictions
        SET actual_default = %s
        WHERE applicant_id = %s
    """

    execute_query(
        query,
        (
            outcome.actual_default,
            applicant_id
        )
    )

    check_query = """
        SELECT COUNT(*)
        FROM predictions
        WHERE applicant_id = %s
    """

    result = execute_query(
        check_query,
        (applicant_id,),
        fetch=True
    )

    if result[0][0] == 0:
        raise HTTPException(
            status_code=404,
            detail="Applicant prediction not found."
        )

    return {
        "applicant_id": applicant_id,
        "actual_default": outcome.actual_default,
        "message": "Actual outcome updated successfully."
    }

@app.get("/monitoring/drift")
def monitoring_drift(days: int = 7):
    return get_feature_drift(days)

@app.get("/monitoring/performance")
def monitoring_performance():
    return get_model_performance()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": model_config["model_version"]
    }


@app.get("/ready")
def readiness_check():
    try:
        execute_query(
            "SELECT 1",
            fetch=True
        )

        return {
            "status": "ready",
            "model_version": model_config["model_version"],
            "database": "ready"
        }

    except DatabaseError:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready."
        )