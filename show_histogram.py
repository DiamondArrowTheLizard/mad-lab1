import calc as c


def show_histogram(
    fig, position, rows, cols, series, color, edgecolor, title, xlabel, ylabel
):
    ax = fig.add_subplot(rows, cols, position)
    if series.dtype == object or str(series.dtype) == "category":
        counts = series.value_counts().sort_index()
        ax.bar(
            counts.index.astype(str), counts.values, color=color, edgecolor=edgecolor
        )
    else:
        bins = c.calc_sturges_rule(series.count())
        ax.hist(series.dropna(), bins=bins, color=color, edgecolor=edgecolor)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
