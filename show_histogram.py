import matplotlib.pyplot as plt

import calc as c

plot_pos = 0


def show_histogram(dataframe, color, edgecolor, title, xlabel, ylabel):
    global plot_pos

    ax = plt.subplot(2, 2, plot_pos + 1)
    plot_pos += 1

    ax.hist(
        dataframe,
        color=color,
        edgecolor=edgecolor,
        bins=c.calc_sturges_rule(dataframe.count()),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
