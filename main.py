from math import log2

import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


def calc_sturges_rule(observations_count):
    return 1 + int(log2(observations_count))


plot_pos = 0


def show_histogram(dataframe, color, edgecolor, title, xlabel, ylabel):
    global plot_pos

    ax = plt.subplot(2, 2, plot_pos + 1)
    plot_pos += 1

    ax.hist(
        dataframe,
        color=color,
        edgecolor=edgecolor,
        bins=calc_sturges_rule(dataframe.count()),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


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

show_histogram(body_style, "blue", "black", "body style", "types", "amount")
show_histogram(fuel_type, "red", "black", "fuel type", "types", "amount")
show_histogram(num_of_cylinders, "orange", "black", "fuel type", "types", "amount")
show_histogram(fuel_system, "purple", "black", "fuel type", "types", "amount")

plt.tight_layout()
plt.show()
