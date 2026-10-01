import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.engineering import engineer_features
from src.config import CONFIG_PATH, MODEL_PATH
from src.database.connection import execute_query
from src.services.prediction_service import generate_prediction
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
    gender: int
    education: int
    marital_status: int
    age: int = Field(ge=18)
    applicant_id: str = Field(min_length=1)

    repay_sep: int = Field(ge=0)
    repay_aug: int = Field(ge=0)
    repay_jul: int = Field(ge=0)
    repay_jun: int = Field(ge=0)
    repay_may: int = Field(ge=0)
    repay_apr: int = Field(ge=0)

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