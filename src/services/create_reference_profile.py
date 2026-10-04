import json

import numpy as np
import pandas as pd


REFERENCE_PATH = (
    "data/processed/credit_card_default_cleaned.csv"
)

OUTPUT_PATH = (
    "data/reference_feature_profile.json"
)

FEATURES = [
    "X1", "X2", "X3", "X4", "X5",
    "X6", "X7", "X8", "X9", "X10", "X11",
    "X12", "X13", "X14", "X15", "X16", "X17",
    "X18", "X19", "X20", "X21", "X22", "X23"
]


def create_reference_profile():
    df = pd.read_csv(REFERENCE_PATH)

    profile = {}

    for feature in FEATURES:
        values = df[feature].dropna()

        bin_edges = np.quantile(
            values,
            np.linspace(0, 1, 11)
        )

        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            profile[feature] = {
                "bin_edges": [],
                "distribution": [1.0]
            }
            continue

        reference_bins = pd.cut(
            values,
            bins=bin_edges,
            include_lowest=True
        )

        distribution = (
            reference_bins
            .value_counts(normalize=True)
            .sort_index()
            .tolist()
        )

        profile[feature] = {
            "bin_edges": [
                float(edge)
                for edge in bin_edges
            ],
            "distribution": [
                float(value)
                for value in distribution
            ]
        }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            profile,
            file,
            indent=2
        )


if __name__ == "__main__":
    create_reference_profile()