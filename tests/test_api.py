from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.database.connection import DatabaseError

from src.services.performance_monitoring_service import (
    get_model_performance,
)

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "model_version" in data


def test_model_info():
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert "model_version" in data
    assert "model_type" in data
    assert "candidate_threshold" in data
    assert "risk_band_boundaries" in data


def test_unknown_applicant():
    response = client.get("/predictions/UNKNOWN-TEST-999")

    assert response.status_code == 404

def test_prediction():
    payload = {
        "applicant_id": "PYTEST-001",
        "credit_limit": 50000,
        "gender": 2,
        "education": 2,
        "marital_status": 1,
        "age": 35,
        "repay_sep": 0,
        "repay_aug": 0,
        "repay_jul": 0,
        "repay_jun": 0,
        "repay_may": 0,
        "repay_apr": 0,
        "bill_sep": 40000,
        "bill_aug": 38000,
        "bill_jul": 35000,
        "bill_jun": 30000,
        "bill_may": 28000,
        "bill_apr": 25000,
        "payment_sep": 5000,
        "payment_aug": 5000,
        "payment_jul": 5000,
        "payment_jun": 5000,
        "payment_may": 5000,
        "payment_apr": 5000
    }

    with patch("src.services.prediction_service.execute_query") as mock_execute_query:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["applicant_id"] == "PYTEST-001"
    assert "default_probability" in data
    assert "predicted_default" in data
    assert "risk_band" in data
    assert "model_version" in data

    assert 0 <= data["default_probability"] <= 1
    assert data["risk_band"] in ["LOW", "MEDIUM", "HIGH"]

    mock_execute_query.assert_called_once()

def test_batch_prediction():
    payload = {
        "applicants": [
            {
                "applicant_id": "BATCH-TEST-001",
                "credit_limit": 50000,
                "gender": 2,
                "education": 2,
                "marital_status": 1,
                "age": 35,
                "repay_sep": 0,
                "repay_aug": 0,
                "repay_jul": 0,
                "repay_jun": 0,
                "repay_may": 0,
                "repay_apr": 0,
                "bill_sep": 40000,
                "bill_aug": 38000,
                "bill_jul": 35000,
                "bill_jun": 30000,
                "bill_may": 28000,
                "bill_apr": 25000,
                "payment_sep": 5000,
                "payment_aug": 5000,
                "payment_jul": 5000,
                "payment_jun": 5000,
                "payment_may": 5000,
                "payment_apr": 5000
            },
            {
                "applicant_id": "BATCH-TEST-002",
                "credit_limit": 100000,
                "gender": 1,
                "education": 1,
                "marital_status": 2,
                "age": 42,
                "repay_sep": 2,
                "repay_aug": 2,
                "repay_jul": 1,
                "repay_jun": 1,
                "repay_may": 0,
                "repay_apr": 0,
                "bill_sep": 90000,
                "bill_aug": 85000,
                "bill_jul": 80000,
                "bill_jun": 75000,
                "bill_may": 70000,
                "bill_apr": 65000,
                "payment_sep": 2000,
                "payment_aug": 3000,
                "payment_jul": 3000,
                "payment_jun": 4000,
                "payment_may": 4000,
                "payment_apr": 5000
            }
        ]
    }

    with patch("src.services.prediction_service.execute_query") as mock_execute_query:
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    assert data["predictions"][0]["applicant_id"] == "BATCH-TEST-001"
    assert data["predictions"][1]["applicant_id"] == "BATCH-TEST-002"

    for prediction in data["predictions"]:
        assert 0 <= prediction["default_probability"] <= 1
        assert prediction["risk_band"] in ["LOW", "MEDIUM", "HIGH"]
        assert "model_version" in prediction

    assert mock_execute_query.call_count == 2

def test_prediction_rejects_negative_credit_limit():
    payload = {
        "applicant_id": "INVALID-001",
        "credit_limit": -50000,
        "gender": 2,
        "education": 2,
        "marital_status": 1,
        "age": 35,
        "repay_sep": 0,
        "repay_aug": 0,
        "repay_jul": 0,
        "repay_jun": 0,
        "repay_may": 0,
        "repay_apr": 0,
        "bill_sep": 40000,
        "bill_aug": 38000,
        "bill_jul": 35000,
        "bill_jun": 30000,
        "bill_may": 28000,
        "bill_apr": 25000,
        "payment_sep": 5000,
        "payment_aug": 5000,
        "payment_jul": 5000,
        "payment_jun": 5000,
        "payment_may": 5000,
        "payment_apr": 5000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_invalid_age():
    payload = {
        "applicant_id": "INVALID-002",
        "credit_limit": 50000,
        "gender": 2,
        "education": 2,
        "marital_status": 1,
        "age": 15,
        "repay_sep": 0,
        "repay_aug": 0,
        "repay_jul": 0,
        "repay_jun": 0,
        "repay_may": 0,
        "repay_apr": 0,
        "bill_sep": 40000,
        "bill_aug": 38000,
        "bill_jul": 35000,
        "bill_jun": 30000,
        "bill_may": 28000,
        "bill_apr": 25000,
        "payment_sep": 5000,
        "payment_aug": 5000,
        "payment_jul": 5000,
        "payment_jun": 5000,
        "payment_may": 5000,
        "payment_apr": 5000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_blank_applicant_id():
    payload = {
        "applicant_id": "",
        "credit_limit": 50000,
        "gender": 2,
        "education": 2,
        "marital_status": 1,
        "age": 35,
        "repay_sep": 0,
        "repay_aug": 0,
        "repay_jul": 0,
        "repay_jun": 0,
        "repay_may": 0,
        "repay_apr": 0,
        "bill_sep": 40000,
        "bill_aug": 38000,
        "bill_jul": 35000,
        "bill_jun": 30000,
        "bill_may": 28000,
        "bill_apr": 25000,
        "payment_sep": 5000,
        "payment_aug": 5000,
        "payment_jul": 5000,
        "payment_jun": 5000,
        "payment_may": 5000,
        "payment_apr": 5000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_database_error():
    with patch(
        "src.services.prediction_service.execute_query",
        side_effect=DatabaseError("Database operation failed.")
    ):
        response = client.post(
            "/predict",
            json={
                "applicant_id": "DB-ERROR-001",
                "credit_limit": 50000,
                "gender": 2,
                "education": 2,
                "marital_status": 1,
                "age": 35,
                "repay_sep": 0,
                "repay_aug": 0,
                "repay_jul": 0,
                "repay_jun": 0,
                "repay_may": 0,
                "repay_apr": 0,
                "bill_sep": 40000,
                "bill_aug": 38000,
                "bill_jul": 35000,
                "bill_jun": 30000,
                "bill_may": 28000,
                "bill_apr": 25000,
                "payment_sep": 5000,
                "payment_aug": 5000,
                "payment_jul": 5000,
                "payment_jun": 5000,
                "payment_may": 5000,
                "payment_apr": 5000,
            }
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Database operation failed."
    }


def test_monitoring_trend():
    response = client.get(
        "/monitoring/trend?days=7"
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    for item in data:
        assert "date" in item
        assert "total_predictions" in item
        assert "predicted_defaults" in item
        assert "predicted_default_rate" in item
        assert "average_probability" in item


def test_monitoring_summary():
    response = client.get(
        "/monitoring/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "predicted_defaults" in data
    assert "predicted_default_rate" in data
    assert "average_probability" in data
    assert "risk_distribution" in data

    assert "LOW" in data["risk_distribution"]
    assert "MEDIUM" in data["risk_distribution"]
    assert "HIGH" in data["risk_distribution"]

def test_monitoring_drift_insufficient_data():
    response = client.get(
        "/monitoring/drift?days=7"
    )

    assert response.status_code == 200

    data = response.json()

    assert "sample_size" in data
    assert "status" in data
    assert "minimum_required" in data
    assert "features" in data

    assert data["status"] == "INSUFFICIENT_DATA"
    assert data["minimum_required"] == 30
    assert data["sample_size"] < 30
    assert data["features"] == []

def test_update_prediction_outcome():
    applicant_id = "DB-TEST-001"

    response = client.patch(
        f"/predictions/{applicant_id}/outcome",
        json={
            "actual_default": True
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["applicant_id"] == applicant_id
    assert data["actual_default"] is True
    assert data["message"] == "Actual outcome updated successfully."


def test_update_prediction_outcome_unknown_applicant():
    response = client.patch(
        "/predictions/UNKNOWN-OUTCOME-001/outcome",
        json={
            "actual_default": False
        }
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Applicant prediction not found."

@app.get("/monitoring/performance")
def monitoring_performance():
    return get_model_performance()

def test_monitoring_performance():
    response = client.get(
        "/monitoring/performance"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_labeled_predictions" in data
    assert "pending_outcomes" in data
    assert "confusion_matrix" in data
    assert "metrics" in data

    assert data["total_labeled_predictions"] >= 1

    assert "true_positive" in data["confusion_matrix"]
    assert "true_negative" in data["confusion_matrix"]
    assert "false_positive" in data["confusion_matrix"]
    assert "false_negative" in data["confusion_matrix"]

    assert "accuracy" in data["metrics"]
    assert "precision" in data["metrics"]
    assert "recall" in data["metrics"]
    assert "f1" in data["metrics"]
    assert "metrics" in data
    assert data["status"] == "INSUFFICIENT_DATA"
    assert data["minimum_required"] == 30


def test_monitoring_performance_counts_false_negative():
    response = client.get(
        "/monitoring/performance"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["confusion_matrix"]["false_negative"] >= 1