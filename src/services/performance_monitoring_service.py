from src.database.connection import execute_query


def get_model_performance() -> dict:
    query = """
        SELECT
            predicted_default,
            actual_default
        FROM predictions
        WHERE actual_default IS NOT NULL
    """

    rows = execute_query(
        query,
        fetch=True
    )

    total_labeled = len(rows)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in rows:
        predicted = row[0]
        actual = row[1]

        if predicted is True and actual is True:
            true_positive += 1

        elif predicted is False and actual is False:
            true_negative += 1

        elif predicted is True and actual is False:
            false_positive += 1

        elif predicted is False and actual is True:
            false_negative += 1

    total_predictions_query = """
        SELECT COUNT(*)
        FROM predictions
    """

    total_predictions = execute_query(
        total_predictions_query,
        fetch=True
    )[0][0]

    pending_outcomes = (
        total_predictions - total_labeled
    )

    accuracy = (
        (true_positive + true_negative)
        / total_labeled
        if total_labeled > 0
        else 0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
    "total_labeled_predictions": total_labeled,
    "pending_outcomes": pending_outcomes,
    "status": (
        "INSUFFICIENT_DATA"
        if total_labeled < 30
        else "OK"
    ),
    "minimum_required": 30,
    "confusion_matrix": {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    },
    "metrics": {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
}