from unittest.mock import patch

from src.api.main import model, model_config
from src.services.prediction_service import (
    generate_prediction,
    get_risk_band,
)


def test_risk_band_low():
    assert get_risk_band(0.10) == "LOW"


def test_risk_band_medium():
    assert get_risk_band(0.40) == "MEDIUM"


def test_risk_band_high():
    assert get_risk_band(0.70) == "HIGH"


def test_generate_prediction():
    applicant_data = {
        "applicant_id": "SERVICE-TEST-001",
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

    with patch(
        "src.services.prediction_service.execute_query"
    ) as mock_execute_query:

        result = generate_prediction(
            applicant_data,
            model,
            model_config,
        )

    assert result["applicant_id"] == "SERVICE-TEST-001"
    assert 0 <= result["default_probability"] <= 1
    assert result["risk_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert "predicted_default" in result
    assert "model_version" in result

    mock_execute_query.assert_called_once()