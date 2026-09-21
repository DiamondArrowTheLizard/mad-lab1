import calc as c

plot_pos = 0


def show_histogram(
    fig, position, rows, cols, dataframe, color, edgecolor, title, xlabel, ylabel
):
    """Draw a histogram on a specific subplot of the given matplotlib Figure.

    fig      : matplotlib.figure.Figure instance
    position : 1-based subplot index
    rows/cols: subplot grid shape
    """
    ax = fig.add_subplot(rows, cols, position)

    ax.hist(
        dataframe,
        color=color,
        edgecolor=edgecolor,
        bins=c.calc_sturges_rule(dataframe.count()),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
