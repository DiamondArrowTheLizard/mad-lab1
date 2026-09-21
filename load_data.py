from ucimlrepo import fetch_ucirepo

import crosstabs as ct


def load_data():
    automobile = fetch_ucirepo(id=10)
    X = automobile.data.features
    y = automobile.data.targets

    print(automobile.variables)

    categorical_vars = [
        "fuel-system",
        "engine-type",
        "drive-wheels",
        "body-style",
        "make",
    ]
    crosstab_list = ct.generate_crosstabs(X, categorical_vars)

    return X, y, crosstab_list
