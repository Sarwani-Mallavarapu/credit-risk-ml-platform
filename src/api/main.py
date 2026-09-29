import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import CONFIG_PATH, MODEL_PATH
from src.database.connection import execute_query
from src.features.engineering import engineer_features

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
    credit_limit: float
    gender: int
    education: int
    marital_status: int
    age: int
    applicant_id: str

    repay_sep: int
    repay_aug: int
    repay_jul: int
    repay_jun: int
    repay_may: int
    repay_apr: int

    bill_sep: float
    bill_aug: float
    bill_jul: float
    bill_jun: float
    bill_may: float
    bill_apr: float

    payment_sep: float
    payment_aug: float
    payment_jul: float
    payment_jun: float
    payment_may: float
    payment_apr: float

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

    input_df = pd.DataFrame([applicant_data])

    input_df = engineer_features(input_df)

    # Model prediction
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

    print(
        f"Prediction saved for applicant: "
        f"{applicant_data['applicant_id']}"
    )
    
    return {
    "applicant_id": applicant_data["applicant_id"],
    "default_probability": round(
        float(default_probability),
        4
    ),
    "predicted_default": bool(predicted_default),
    "risk_band": risk_band,
    "model_version": model_config["model_version"]
}


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

        input_df = pd.DataFrame([applicant_data])
        input_df = engineer_features(input_df)

        default_probability = model.predict_proba(input_df)[:, 1][0]

        candidate_threshold = model_config["candidate_threshold"]
        predicted_default = default_probability >= candidate_threshold
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

        results.append({
            "applicant_id": applicant_data["applicant_id"],
            "default_probability": round(float(default_probability), 4),
            "predicted_default": bool(predicted_default),
            "risk_band": risk_band,
            "model_version": model_config["model_version"]
        })

    return {
        "count": len(results),
        "predictions": results
    }