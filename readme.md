# Credit Risk ML Platform

An end-to-end machine learning platform for **credit default prediction, risk classification, model explainability, and API-based inference**.

The project takes applicant credit and repayment data, performs feature engineering, applies a trained Random Forest model, exposes predictions through a FastAPI service, and persists prediction results in PostgreSQL.

---

## Overview

The **Credit Risk ML Platform** demonstrates a production-oriented workflow for deploying a machine learning model beyond a notebook.

The project covers:

* Exploratory data analysis and data validation
* Feature engineering
* Baseline model comparison
* Random Forest hyperparameter tuning
* Classification threshold analysis
* SHAP-based model explainability
* Model serialization and configuration management
* FastAPI-based inference
* PostgreSQL prediction persistence
* Batch prediction
* Automated API testing

### End-to-End Flow

```text
Applicant Data
      │
      ▼
Feature Engineering
      │
      ▼
Trained Random Forest Model
      │
      ├──► Default Probability
      │
      ├──► Default Classification
      │
      └──► Risk Band
              │
              ▼
          FastAPI API
              │
              ▼
          PostgreSQL
```

---

## Machine Learning

### Dataset

The project uses the **UCI Credit Card Default dataset**, containing 30,000 applicant records.

Target distribution:

| Target     | Records |
| ---------- | ------: |
| No Default |  23,364 |
| Default    |   6,636 |

The dataset contains applicant demographic information, credit limits, repayment status, bill amounts, and payment amounts.

---

## Feature Engineering

Additional features were created to capture repayment behavior, bill/payment patterns, and credit utilization.

Examples include:

* `avg_bill_amount`
* `max_bill_amount`
* `min_bill_amount`
* `total_bill_amount`
* `bill_amount_std`
* `max_repayment_status`
* `avg_repayment_status`
* `months_serious_delay`
* `months_with_delay`
* `recent_repayment_status`
* `total_payment_amount`
* `avg_payment_amount`
* `max_payment_amount`
* `payment_to_bill_ratio`
* `age_group`

The feature engineering logic is implemented as a reusable function in:

```text
src/features/engineering.py
```

This allows the same transformations to be applied during API inference.

---

## Model Development

Multiple baseline classifiers were evaluated before selecting the final model.

### Baseline Results

| Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | ----: | ------: | -----: |
| Random Forest       |    0.818 |     0.663 |  0.362 | 0.468 |   0.779 |  0.555 |
| Decision Tree       |    0.816 |     0.664 |  0.339 | 0.449 |   0.759 |  0.510 |
| Logistic Regression |    0.812 |     0.662 |  0.307 | 0.419 |   0.754 |  0.524 |

Random Forest was subsequently tuned using `RandomizedSearchCV` with 5-fold cross-validation and ROC-AUC as the optimization metric.

### Tuned Random Forest

Selected configuration:

```text
n_estimators      = 400
max_depth         = 8
max_features      = None
min_samples_split = 20
min_samples_leaf  = 20
class_weight      = balanced
```

Cross-validation ROC-AUC:

```text
0.7857
```

### Test Set Performance

```text
Accuracy   : 0.7722
Precision  : 0.4878
Recall     : 0.6014
F1 Score   : 0.5386
ROC-AUC    : 0.7797
PR-AUC     : 0.5527
```

The project also evaluates classification thresholds separately from the model's probability output. A candidate threshold of **0.55** was selected for the current API configuration based on the project’s threshold analysis.

---

## Risk Classification

The API converts the model's default probability into configurable risk bands.

| Probability     | Risk Band |
| --------------- | --------- |
| `< 0.30`        | LOW       |
| `0.30 – < 0.55` | MEDIUM    |
| `≥ 0.55`        | HIGH      |

These boundaries are **project-defined configuration values**, not industry-standard credit-risk thresholds.

The configuration is stored separately from the model:

```text
models/model_config.joblib
```

---

## Model Explainability

SHAP was used to analyze feature importance and understand which features contribute most to the model's predictions.

Important features identified during analysis include:

* `max_repayment_status`
* `months_with_delay`
* `credit_limit`
* `months_serious_delay`
* `repay_sep`
* `recent_repayment_status`
* `payment_to_bill_ratio`
* `bill_amount_std`
* `bill_sep`
* `max_payment_amount`

SHAP importance indicates the magnitude of contribution to model predictions; it should not be interpreted as a causal relationship.

---

## API

The model is exposed through a FastAPI application.

### Available Endpoints

| Method | Endpoint                      | Purpose                              |
| ------ | ----------------------------- | ------------------------------------ |
| GET    | `/health`                     | API health check                     |
| GET    | `/model/info`                 | Model and configuration information  |
| POST   | `/predict`                    | Predict risk for one applicant       |
| POST   | `/predict/batch`              | Predict risk for multiple applicants |
| GET    | `/predictions/{applicant_id}` | Retrieve stored predictions          |

Interactive API documentation is automatically available through Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Example Prediction

### Request

```json
{
  "applicant_id": "SWAGGER-001",
  "credit_limit": 50000,
  "gender": 2,
  "education": 2,
  "marital_status": 1,
  "age": 30,
  "repay_sep": 0,
  "repay_aug": 0,
  "repay_jul": 0,
  "repay_jun": 0,
  "repay_may": 0,
  "repay_apr": 0,
  "bill_sep": 20000,
  "bill_aug": 18000,
  "bill_jul": 15000,
  "bill_jun": 12000,
  "bill_may": 10000,
  "bill_apr": 8000,
  "payment_sep": 5000,
  "payment_aug": 5000,
  "payment_jul": 4000,
  "payment_jun": 4000,
  "payment_may": 3000,
  "payment_apr": 3000
}
```

### Response

```json
{
  "applicant_id": "SWAGGER-001",
  "default_probability": 0.XXXX,
  "predicted_default": false,
  "risk_band": "LOW",
  "model_version": "v1.0"
}
```

The probability and risk classification are generated by the trained model and configured threshold.

---

## PostgreSQL Persistence

Prediction results are stored in PostgreSQL.

Database:

```text
credit_risk
```

Table:

```text
predictions
```

Stored information includes:

* Applicant ID
* Default probability
* Predicted default
* Risk band
* Model version
* Prediction timestamp

Database configuration is loaded through environment variables rather than hard-coded credentials.

Example:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk
DB_USER=postgres
DB_PASSWORD=<your-password>
```

The `.env` file is excluded from version control.

---

## Project Structure

```text
credit-risk-ml-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│   ├── credit_risk_model.joblib
│   └── model_config.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   │
│   └── database/
│       ├── __init__.py
│       └── connection.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── .env
├── pytest.ini
├── requirements.txt
└── README.md
```

> Model artifacts, raw data, environment files, and virtual environments are excluded from Git where appropriate.

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd credit-risk-ml-platform
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a PostgreSQL database named:

```text
credit_risk
```

Create the `predictions` table using the SQL schema defined for the project.

### 5. Configure environment variables

Create `.env` in the project root:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk
DB_USER=postgres
DB_PASSWORD=<your-password>
```

---

## Running the API

From the project root:

```powershell
python -m uvicorn src.api.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Running Tests

Run the automated API tests with:

```powershell
pytest
```

The current test suite covers:

* Health endpoint
* Model information endpoint
* Missing applicant handling
* Single prediction
* Batch prediction

---

## Technology Stack

### Programming & Data

* Python
* Pandas
* NumPy
* SciPy
* Scikit-learn

### Machine Learning

* Random Forest
* RandomizedSearchCV
* SHAP
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Database

* PostgreSQL
* psycopg2

### Testing

* Pytest
* FastAPI TestClient

### Development

* Jupyter Notebook
* Git
* Python virtual environments

---

## Future Improvements

Potential extensions include:

* Docker containerization
* CI/CD pipeline
* Model monitoring
* Prediction drift monitoring
* Feature drift detection
* Authentication and authorization
* Centralized logging
* Model version registry
* Cloud deployment
* Production database connection pooling
* Explainability endpoint for individual predictions

---

## Project Status

**Current status: Functional end-to-end ML API**

The current implementation supports model inference, configurable risk classification, batch prediction, PostgreSQL persistence, Swagger-based API interaction, and automated testing.
