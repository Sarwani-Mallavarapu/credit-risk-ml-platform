# Credit Risk ML Platform

An end-to-end machine learning platform for **credit default prediction, risk classification, model explainability, and API-based inference**.

The platform combines a trained Random Forest model with a FastAPI backend, PostgreSQL persistence, Docker-based deployment, and monitoring capabilities for prediction trends, feature drift, and model performance.

**Live API:** https://credit-risk-ml-platform.onrender.com  
**Interactive API Documentation:** https://credit-risk-ml-platform.onrender.com/docs  
**GitHub Repository:** https://github.com/Sarwani-Mallavarapu/credit-risk-ml-platform

---

## Overview

The Credit Risk ML Platform demonstrates how a machine learning model can be developed, evaluated, deployed, and monitored through a backend API.

The project covers the machine learning lifecycle, from exploratory data analysis and feature engineering to model evaluation, API inference, database persistence, and outcome-based performance monitoring.

### Key Features

- Exploratory data analysis and data validation
- Feature engineering for credit and repayment behavior
- Baseline model comparison and hyperparameter tuning
- Classification threshold analysis
- SHAP-based model explainability
- Single and batch prediction APIs
- Configurable risk classification
- PostgreSQL prediction persistence
- Actual-outcome recording for model evaluation
- Prediction summary and trend monitoring
- Feature drift monitoring
- Outcome-based performance monitoring
- Automated API testing
- Docker containerization and cloud deployment on Render

## Architecture

```text
               Applicant Credit Data
                        |
                        v
                Feature Engineering
                        |
                        v
               Random Forest Model
                        |
                        v
               Default Probability
                        |
                        v
             Classification Threshold
                        |
                        v
                Risk Classification
                        |
                        v
                  FastAPI Service
                        |
              +---------+---------+
              |                   |
              v                   v
       PostgreSQL DB       Monitoring APIs
              |                   |
              v                   v
       Stored Predictions   Trends / Drift /
                            Performance
              |
              v
       Actual Outcome Feedback
```

The API exposes model inference and monitoring functionality, while PostgreSQL stores prediction records and associated actual outcomes.

---

## Machine Learning

### Dataset

The project uses the **UCI Credit Card Default dataset**, containing 30,000 records.

| Target | Records |
|---|---:|
| No Default | 23,364 |
| Default | 6,636 |
| **Total** | **30,000** |

The dataset includes demographic attributes, credit limits, repayment status, bill amounts, and payment amounts.

### Feature Engineering

Additional features were engineered to represent repayment behavior, payment patterns, and credit utilization.

Examples include:

- `avg_bill_amount`
- `max_bill_amount`
- `min_bill_amount`
- `total_bill_amount`
- `bill_amount_std`
- `max_repayment_status`
- `avg_repayment_status`
- `months_serious_delay`
- `months_with_delay`
- `recent_repayment_status`
- `total_payment_amount`
- `avg_payment_amount`
- `max_payment_amount`
- `payment_to_bill_ratio`
- `age_group`

The feature engineering logic is implemented in:

```text
src/features/engineering.py
```

The same feature transformations are applied during inference to maintain consistency between model training and API predictions.

---

## Model Development and Evaluation

Multiple classifiers were evaluated before selecting Random Forest as the final model.

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.818 | 0.663 | 0.362 | 0.468 | 0.779 | 0.555 |
| Decision Tree | 0.816 | 0.664 | 0.339 | 0.449 | 0.759 | 0.510 |
| Logistic Regression | 0.812 | 0.662 | 0.307 | 0.419 | 0.754 | 0.524 |

### Hyperparameter Tuning

Random Forest was tuned using `RandomizedSearchCV` with five-fold cross-validation and ROC-AUC as the optimization metric.

Selected configuration:

```text
n_estimators      = 400
max_depth         = 8
max_features      = None
min_samples_split = 20
min_samples_leaf  = 20
class_weight      = balanced
```

Cross-validation ROC-AUC: **0.7857**

### Tuned Model Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7722 |
| Precision | 0.4878 |
| Recall | 0.6014 |
| F1 Score | 0.5386 |
| ROC-AUC | 0.7797 |
| PR-AUC | 0.5527 |

A separate threshold analysis evaluated the trade-off between precision and recall.

At the candidate threshold of `0.55`, the model achieved:

| Metric | Score |
|---|---:|
| Precision | 0.5253 |
| Recall | 0.5629 |
| F1 Score | 0.5435 |

The threshold is a project-level decision based on the evaluation results, rather than a universal credit-risk standard.

---

## Risk Classification

The API converts the predicted default probability into one of three configurable risk bands.

| Default Probability | Risk Band |
|---|---|
| Below `0.30` | LOW |
| `0.30` to below `0.55` | MEDIUM |
| `0.55` or higher | HIGH |

The default classification threshold is `0.55`.

These boundaries are project-defined configuration values and should not be interpreted as industry-standard lending decisions.

The deployed model information endpoint exposes the model version, threshold, and risk-band boundaries.

**Endpoint:** `/model/info`

---

## Model Explainability

SHAP was used to examine feature contributions to model predictions.

Important features identified during analysis include:

- `max_repayment_status`
- `months_with_delay`
- `credit_limit`
- `months_serious_delay`
- `repay_sep`
- `recent_repayment_status`
- `payment_to_bill_ratio`
- `bill_amount_std`
- `bill_sep`
- `max_payment_amount`

SHAP helps explain model behavior, but feature importance does not establish causation. Explanations should be interpreted in the context of the dataset, model, and intended use.

---

## REST API

The model is served through FastAPI, with interactive documentation available through Swagger UI.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| GET | `/ready` | Check application readiness |
| GET | `/model/info` | Retrieve model version and classification configuration |
| POST | `/predict` | Predict default probability and risk for one applicant |
| POST | `/predict/batch` | Generate predictions for multiple applicants |
| GET | `/predictions/{applicant_id}` | Retrieve a stored prediction |
| PATCH | `/predictions/{applicant_id}/outcome` | Record an applicant's actual default outcome |
| GET | `/monitoring/summary` | Summarize prediction volume and risk distribution |
| GET | `/monitoring/summary/period` | Retrieve prediction summaries for a period |
| GET | `/monitoring/trend` | Inspect prediction trends over time |
| GET | `/monitoring/drift` | Evaluate feature drift against a reference profile |
| GET | `/monitoring/performance` | Evaluate predictions against recorded outcomes |

### Example Prediction Request

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

### Example Prediction Response

The following is an illustrative response; the actual probability and risk band depend on model inference.

```json
{
  "applicant_id": "SWAGGER-001",
  "default_probability": 0.24,
  "predicted_default": false,
  "risk_band": "LOW",
  "model_version": "v1.0"
}
```

### Outcome Feedback

After an applicant's actual outcome becomes available, it can be recorded through:

```text
PATCH /predictions/{applicant_id}/outcome
```

This supports subsequent comparison of predicted classifications against actual outcomes.

Use the live Swagger documentation to inspect the required request schema and test each endpoint.

---

## Monitoring and Model Performance

The platform includes monitoring endpoints to help evaluate API usage, prediction behavior, feature drift, and model performance.

### Prediction Summary

`GET /monitoring/summary`

Reports prediction statistics such as:

- Total predictions
- Predicted defaults
- Predicted default rate
- Average predicted probability
- Risk-band distribution

### Prediction Trends

`GET /monitoring/trend`

Provides access to prediction trends over time.

`GET /monitoring/summary/period`

Provides prediction summaries for a specified period.

### Feature Drift

`GET /monitoring/drift`

Compares observed prediction features with a reference feature profile.

The endpoint can return `INSUFFICIENT_DATA` when the sample size is below the configured minimum. In the current implementation, the minimum is 30 observations.

### Outcome-Based Performance

`GET /monitoring/performance`

Evaluates predictions against recorded actual outcomes and reports:

- Total labeled predictions
- Pending outcomes
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 score
- Evaluation status

The current implementation requires at least 30 labeled predictions before reporting sufficient-data status.

When no actual outcomes have been recorded, the metrics are not meaningful indicators of model quality. A sufficiently large and representative set of labeled outcomes is needed for a reliable evaluation.

---

## PostgreSQL Persistence

Prediction records are stored in PostgreSQL.

**Database:** `credit_risk`  
**Table:** `predictions`

Stored information includes:

- Applicant ID
- Default probability
- Predicted default classification
- Risk band
- Model version
- Prediction timestamp
- Actual outcome, when available

Database connection settings are supplied through environment variables rather than hard-coded credentials.

Example local configuration:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk
DB_USER=postgres
DB_PASSWORD=<your-password>
```

Create the database and initialize the schema required by the application before running the API locally. Keep real credentials in a local `.env` file or environment variables, and never commit secrets to Git.

---

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy, SciPy |
| Machine Learning | Scikit-learn, Random Forest, RandomizedSearchCV |
| Explainability | SHAP |
| Model Artifacts | Joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL, psycopg2 |
| Testing | Pytest, FastAPI TestClient |
| Containerization | Docker |
| Deployment | Render |
| Development | Jupyter Notebook, Git, PowerShell |
| CI | GitHub Actions |

---

## Project Structure

```text
credit-risk-ml-platform/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference_feature_profile.json
├── models/
│   ├── credit_risk_model.joblib
│   └── model_config.joblib
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py
│   └── services/
│       └── monitoring_service.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

This is a representative project structure; additional files may exist in the repository.

Large datasets, virtual environments, environment files, and generated model artifacts should be excluded from version control when appropriate. The deployed Docker image must include the model artifacts and reference profile required at runtime.

---

## Local Setup

### Prerequisites

- Python 3.10
- PostgreSQL
- Git
- The model artifacts required by the application

### 1. Clone the repository

```powershell
git clone https://github.com/Sarwani-Mallavarapu/credit-risk-ml-platform.git
cd credit-risk-ml-platform
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a database named `credit_risk` and initialize the schema expected by the application.

Set the database environment variables for your local environment.

### 5. Verify required artifacts

Ensure that the model files expected by the application are present in the appropriate `models/` directory, along with `data/reference_feature_profile.json`.

If model artifacts are excluded from Git, obtain them from the project's trusted artifact source before starting the API.

### 6. Start the API

From the repository root:

```powershell
python -m uvicorn src.api.main:app --reload
```

Open the local API documentation:

http://127.0.0.1:8000/docs

---

## Running Tests

Run the test suite from the repository root:

```powershell
python -m pytest
```

The tests cover API behavior, including health checks, model information, applicant handling, single predictions, and batch predictions.

The previously recorded local test run passed 22 tests. Run the current suite to verify the latest repository state.

---

## Docker and Deployment

The application is containerized using Docker and deployed on Render.

The Docker image installs the Python dependencies, copies the application source code and required runtime assets, and launches the FastAPI application with Uvicorn.

The service reads its port from the deployment environment and uses the configured port, defaulting to `10000`.

### Live Resources

- **API:** https://credit-risk-ml-platform.onrender.com
- **Swagger UI:** https://credit-risk-ml-platform.onrender.com/docs
- **Health check:** https://credit-risk-ml-platform.onrender.com/health
- **Readiness check:** https://credit-risk-ml-platform.onrender.com/ready
- **Model information:** https://credit-risk-ml-platform.onrender.com/model/info
- **Prediction summary:** https://credit-risk-ml-platform.onrender.com/monitoring/summary?days=7
- **Feature drift:** https://credit-risk-ml-platform.onrender.com/monitoring/drift?days=7
- **Performance monitoring:** https://credit-risk-ml-platform.onrender.com/monitoring/performance?days=30

Monitoring endpoints can return insufficient-data statuses until enough predictions or labeled outcomes are available. A successful HTTP response does not, by itself, establish model quality or production readiness.

---

## Future Improvements

Potential next steps include:

- Authentication and authorization
- Structured logging and centralized observability
- Database connection pooling and resilience improvements
- Individual prediction explainability through the API
- Model and data version tracking
- More comprehensive integration and deployment tests
- Automated monitoring alerts
- Evaluation using a larger set of actual outcomes
- Additional deployment and operational safeguards

---

## Project Status

**Status: Deployed end-to-end machine learning API with prediction storage and monitoring capabilities.**

The current implementation supports:

- Trained Random Forest inference
- Configurable default classification and risk bands
- Single and batch predictions
- PostgreSQL persistence
- Actual-outcome recording
- Prediction summaries and trends
- Feature drift monitoring
- Outcome-based performance evaluation
- Docker-based deployment on Render
- Automated API tests

The project demonstrates practical experience across machine learning, feature engineering, Python backend development, API design, relational database integration, deployment, testing, and model monitoring.

The monitoring workflow is implemented, but reliable production performance evaluation still depends on collecting sufficient representative actual outcomes.