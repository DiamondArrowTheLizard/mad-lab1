import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

import show_histogram as sh

if __name__ == "__main__":
    # fetch dataset
    automobile = fetch_ucirepo(id=10)

    # data (as pandas dataframes)
    X = automobile.data.features
    y = automobile.data.targets

    # metadata
    # print(automobile.metadata)

    # variable information
    print(automobile.variables)

    body_style = X["body-style"]
    fuel_type = X["fuel-type"]
    num_of_cylinders = X["num-of-cylinders"]
    fuel_system = X["fuel-system"]

    sh.show_histogram(body_style, "blue", "black", "body style", "types", "amount")
    sh.show_histogram(fuel_type, "red", "black", "fuel type", "types", "amount")
    sh.show_histogram(
        num_of_cylinders, "orange", "black", "fuel type", "types", "amount"
    )
    sh.show_histogram(fuel_system, "purple", "black", "fuel system", "types", "amount")

    plt.tight_layout()
    plt.show()
