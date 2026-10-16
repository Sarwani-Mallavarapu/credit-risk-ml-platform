
import logging
import time

import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import CONFIG_PATH, MODEL_PATH
from src.database.connection import DatabaseError, execute_query
from src.logging_config import configure_logging
from src.services.monitoring_service import (
    get_feature_drift,
    get_prediction_summary,
    get_prediction_summary_by_period,
    get_prediction_trend,
)
from src.services.performance_monitoring_service import get_model_performance
from src.services.prediction_service import generate_prediction


# --------------------------------------------------
# Logging
# --------------------------------------------------

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
    version=model_config["model_version"],
)


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class ApplicantRequest(BaseModel):
    applicant_id: str = Field(min_length=1, max_length=100)

    credit_limit: float = Field(ge=0)
    gender: int = Field(ge=1, le=2)
    education: int = Field(ge=0, le=6)
    marital_status: int = Field(ge=0, le=3)
    age: int = Field(ge=18, le=100)

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
    applicants: list[ApplicantRequest] = Field(
        min_length=1,
        max_length=100,
    )


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
    return "HIGH"


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": model_config["model_version"],
    }


# --------------------------------------------------
# Readiness endpoint
# --------------------------------------------------

@app.get("/ready")
def readiness_check():
    try:
        execute_query("SELECT 1", fetch=True)

        return {
            "status": "ready",
            "model_version": model_config["model_version"],
            "database": "ready",
        }

    except DatabaseError:
        logger.exception("Database readiness check failed")
        raise HTTPException(
            status_code=503,
            detail="Service is not ready.",
        )


# --------------------------------------------------
# Model information endpoint
# --------------------------------------------------

@app.get("/model/info")
def model_info():
    return {
        "model_version": model_config["model_version"],
        "model_type": model_config["model_type"],
        "candidate_threshold": model_config["candidate_threshold"],
        "risk_band_boundaries": model_config["risk_band_boundaries"],
    }


# --------------------------------------------------
# Single prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(applicant: ApplicantRequest):
    applicant_data = applicant.model_dump()

    return generate_prediction(
        applicant_data,
        model,
        model_config,
    )


# --------------------------------------------------
# Batch prediction endpoint
# --------------------------------------------------

@app.post("/predict/batch")
def predict_batch(batch: BatchApplicantRequest):
    results = []

    for applicant in batch.applicants:
        prediction = generate_prediction(
            applicant.model_dump(),
            model,
            model_config,
        )
        results.append(prediction)

    return {
        "count": len(results),
        "predictions": results,
    }


# --------------------------------------------------
# Retrieve applicant predictions
# --------------------------------------------------

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
        ORDER BY created_at DESC, id DESC
    """

    results = execute_query(
        query,
        (applicant_id,),
        fetch=True,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for applicant: {applicant_id}",
        )

    predictions = []

    for row in results:
        predictions.append({
            "id": row[0],
            "applicant_id": row[1],
            "default_probability": float(row[2]),
            "predicted_default": bool(row[3]),
            "risk_band": row[4],
            "model_version": row[5],
            "created_at": (
                row[6].isoformat()
                if row[6] is not None
                else None
            ),
        })

    return {
        "applicant_id": applicant_id,
        "count": len(predictions),
        "predictions": predictions,
    }


# --------------------------------------------------
# Update actual outcome
# --------------------------------------------------

@app.patch("/predictions/{applicant_id}/outcome")
def update_prediction_outcome(
    applicant_id: str,
    outcome: OutcomeRequest,
):
    # Check whether predictions exist before updating.
    check_query = """
        SELECT COUNT(*)
        FROM predictions
        WHERE applicant_id = %s
    """

    result = execute_query(
        check_query,
        (applicant_id,),
        fetch=True,
    )

    if not result or result[0][0] == 0:
        raise HTTPException(
            status_code=404,
            detail="Applicant prediction not found.",
        )

    # Updates all prediction records for this applicant.
    update_query = """
        UPDATE predictions
        SET actual_default = %s
        WHERE applicant_id = %s
    """

    execute_query(
        update_query,
        (outcome.actual_default, applicant_id),
    )

    return {
        "applicant_id": applicant_id,
        "actual_default": outcome.actual_default,
        "message": "Actual outcome updated successfully.",
    }


# --------------------------------------------------
# Monitoring endpoints
# --------------------------------------------------

@app.get("/monitoring/summary")
def monitoring_summary():
    return get_prediction_summary()


@app.get("/monitoring/summary/period")
def monitoring_summary_period(
    days: int = Query(default=7, ge=1, le=365),
):
    return get_prediction_summary_by_period(days)


@app.get("/monitoring/trend")
def monitoring_trend(
    days: int = Query(default=7, ge=1, le=365),
):
    return get_prediction_trend(days)


@app.get("/monitoring/drift")
def monitoring_drift(
    days: int = Query(default=7, ge=1, le=365),
):
    return get_feature_drift(days)


@app.get("/monitoring/performance")
def monitoring_performance():
    return get_model_performance()


# --------------------------------------------------
# Database exception handler
# --------------------------------------------------

@app.exception_handler(DatabaseError)
async def database_exception_handler(
    request: Request,
    exc: DatabaseError,
):
    logger.error(
        "Database operation failed on path=%s: %s",
        request.url.path,
        type(exc).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database operation failed.",
        },
    )


# --------------------------------------------------
# Request logging middleware
# --------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "HTTP request failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "HTTP request method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response