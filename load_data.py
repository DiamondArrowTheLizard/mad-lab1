import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

import anova as anv
import crosstabs as ct
import describes as dsc
import regression as reg
import show_correlation as sc

CATEGORICAL_VARS = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make"]


def _coerce_and_clean(df):
    df = df.copy().replace("?", np.nan)
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted
    return df


def _resolve_target(features, targets):
    y = targets if targets is not None else pd.DataFrame()
    if "price" not in y.columns and "price" in features.columns:
        y = features[["price"]].copy()
    if not y.empty:
        features = features.drop(
            columns=[c for c in y.columns if c in features.columns]
        )
    return features, y


def load_data():
    automobile = fetch_ucirepo(id=10)
    X, y = _resolve_target(automobile.data.features, automobile.data.targets)
    X = _coerce_and_clean(X)
    y = _coerce_and_clean(y)

    combined = pd.concat([X, y], axis=1) if not y.empty else X

    available_cat = [c for c in CATEGORICAL_VARS if c in X.columns]
    crosstab_list = ct.generate_crosstabs(X, available_cat)
    describe_list = dsc.generate_describes(combined)
    correlation_list = sc.generate_correlations(combined)
    regression_models = reg.build_models(combined)
    anova_results = anv.generate_anova(combined)

    return (
        X,
        y,
        crosstab_list,
        correlation_list,
        describe_list,
        regression_models,
        anova_results,
    )
