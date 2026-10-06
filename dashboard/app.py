import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Credit Risk Monitoring",
    page_icon="📊",
    layout="wide"
)


st.title("Credit Risk Monitoring Dashboard")

st.markdown(
    "Monitor prediction activity, feature drift, and model performance."
)


try:
    summary_response = requests.get(
        f"{API_BASE_URL}/monitoring/summary",
        timeout=5
    )

    summary_response.raise_for_status()

    summary = summary_response.json()

except requests.RequestException:
    st.error(
        "Unable to connect to the Credit Risk API. "
        "Make sure FastAPI is running on port 8000."
    )
    st.stop()


# --------------------------------------------------
# Prediction Overview
# --------------------------------------------------

st.subheader("Prediction Overview")


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    summary["total_predictions"]
)

col2.metric(
    "Predicted Defaults",
    summary["predicted_defaults"]
)

col3.metric(
    "Default Rate",
    f'{summary["predicted_default_rate"]:.2%}'
)

col4.metric(
    "Average Probability",
    f'{summary["average_probability"]:.2%}'
)


# --------------------------------------------------
# Risk Distribution
# --------------------------------------------------

st.subheader("Risk Distribution")


risk_distribution = summary["risk_distribution"]

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric(
    "LOW Risk",
    risk_distribution["LOW"]
)

risk_col2.metric(
    "MEDIUM Risk",
    risk_distribution["MEDIUM"]
)

risk_col3.metric(
    "HIGH Risk",
    risk_distribution["HIGH"]
)


# --------------------------------------------------
# Prediction Trend
# --------------------------------------------------

st.subheader("Prediction Trend")


try:
    trend_response = requests.get(
        f"{API_BASE_URL}/monitoring/trend",
        params={"days": 7},
        timeout=5
    )

    trend_response.raise_for_status()

    trend = trend_response.json()

except requests.RequestException:
    st.error("Unable to load prediction trend.")
    trend = []


if trend:

    trend_data = {
        "date": [
            item["date"]
            for item in trend
        ],
        "predictions": [
            item["total_predictions"]
            for item in trend
        ]
    }

    st.line_chart(
        trend_data,
        x="date",
        y="predictions"
    )

else:

    st.info(
        "No prediction trend data available for the selected period."
    )

# --------------------------------------------------
# Feature Drift
# --------------------------------------------------

st.subheader("Feature Drift")

try:
    drift_response = requests.get(
        f"{API_BASE_URL}/monitoring/drift",
        params={"days": 7},
        timeout=5
    )

    drift_response.raise_for_status()

    drift = drift_response.json()

except requests.RequestException:
    st.error("Unable to load feature drift data.")
    drift = None


if drift:

    if drift["status"] == "INSUFFICIENT_DATA":

        st.warning(
            f'Insufficient data for drift monitoring. '
            f'Current sample: {drift["sample_size"]}. '
            f'Minimum required: {drift["minimum_required"]}.'
        )

    elif drift["status"] == "OK":

        drift_rows = []

        for feature in drift["features"]:
            drift_rows.append(
                {
                    "Feature": feature["feature"],
                    "PSI": feature["psi"],
                    "Drift Level": feature["drift_level"]
                }
            )

        st.dataframe(
            drift_rows,
            use_container_width=True
        )

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("Model Performance")

try:
    performance_response = requests.get(
        f"{API_BASE_URL}/monitoring/performance",
        timeout=5
    )

    performance_response.raise_for_status()

    performance = performance_response.json()

except requests.RequestException:
    st.error("Unable to load model performance data.")
    performance = None


if performance:

    if performance["status"] == "INSUFFICIENT_DATA":

        st.warning(
            f'Insufficient labeled data for reliable model '
            f'performance evaluation. '
            f'Current labeled outcomes: '
            f'{performance["total_labeled_predictions"]}. '
            f'Minimum required: '
            f'{performance["minimum_required"]}.'
        )

    metrics = performance["metrics"]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        "Accuracy",
        f'{metrics["accuracy"]:.2%}'
    )

    metric_col2.metric(
        "Precision",
        f'{metrics["precision"]:.2%}'
    )

    metric_col3.metric(
        "Recall",
        f'{metrics["recall"]:.2%}'
    )

    metric_col4.metric(
        "F1",
        f'{metrics["f1"]:.2%}'
    )

    st.markdown("### Confusion Matrix")

    confusion = performance["confusion_matrix"]

    matrix_data = {
        "Actual Negative": [
            confusion["true_negative"],
            confusion["false_positive"]
        ],
        "Actual Positive": [
            confusion["false_negative"],
            confusion["true_positive"]
        ]
    }

    st.dataframe(
    {
        "Prediction": [
            "Predicted Negative",
            "Predicted Positive"
        ],
        "Actual Negative": [
            confusion["true_negative"],
            confusion["false_positive"]
        ],
        "Actual Positive": [
            confusion["false_negative"],
            confusion["true_positive"]
        ]
    },
    use_container_width=True,
    hide_index=True
)