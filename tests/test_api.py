from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


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

    with patch("src.api.main.execute_query") as mock_execute_query:
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

    with patch("src.api.main.execute_query") as mock_execute_query:
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