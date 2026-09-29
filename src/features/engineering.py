import numpy as np
import pandas as pd


BILL_COLUMNS = [
    "bill_sep",
    "bill_aug",
    "bill_jul",
    "bill_jun",
    "bill_may",
    "bill_apr"
]

REPAYMENT_COLUMNS = [
    "repay_sep",
    "repay_aug",
    "repay_jul",
    "repay_jun",
    "repay_may",
    "repay_apr"
]

PAYMENT_COLUMNS = [
    "payment_sep",
    "payment_aug",
    "payment_jul",
    "payment_jun",
    "payment_may",
    "payment_apr"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bill amount features
    df["avg_bill_amount"] = df[BILL_COLUMNS].mean(axis=1)
    df["max_bill_amount"] = df[BILL_COLUMNS].max(axis=1)
    df["min_bill_amount"] = df[BILL_COLUMNS].min(axis=1)

    # Repayment behavior features
    df["max_repayment_status"] = (
        df[REPAYMENT_COLUMNS].max(axis=1)
    )

    df["avg_repayment_status"] = (
        df[REPAYMENT_COLUMNS].mean(axis=1)
    )

    df["months_serious_delay"] = (
        df[REPAYMENT_COLUMNS] >= 2
    ).sum(axis=1)

    df["months_with_delay"] = (
        df[REPAYMENT_COLUMNS] > 0
    ).sum(axis=1)

    df["recent_repayment_status"] = (
        df["repay_sep"]
    )

    # Bill amount features
    df["total_bill_amount"] = (
        df[BILL_COLUMNS].sum(axis=1)
    )

    df["bill_amount_std"] = (
        df[BILL_COLUMNS].std(axis=1)
    )

    # Payment features
    df["total_payment_amount"] = (
        df[PAYMENT_COLUMNS].sum(axis=1)
    )

    df["avg_payment_amount"] = (
        df[PAYMENT_COLUMNS].mean(axis=1)
    )

    df["max_payment_amount"] = (
        df[PAYMENT_COLUMNS].max(axis=1)
    )

    # Payment-to-bill ratio
    df["payment_to_bill_ratio"] = (
        df["total_payment_amount"]
        / df["total_bill_amount"].replace(0, np.nan)
    )

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[20, 30, 40, 50, 60, 80],
        labels=[
            "21-29",
            "30-39",
            "40-49",
            "50-59",
            "60+"
        ],
        right=False
    )

    return df