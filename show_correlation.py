import seaborn as sns


class Correlation:
    def __init__(self, name: str, data):
        self.name = name
        self.data = data

    def __str__(self):
        return f"Correlation ({self.name})\n{self.data}"


def get_correlation_pearson(dataframe):
    return dataframe.corr(method="pearson")


def get_correlation_spearman(dataframe):
    return dataframe.corr(method="spearman")


def get_correlation_kendall(dataframe):
    return dataframe.corr(method="kendall")


def generate_correlations(dataframe) -> list:
    numeric_df = dataframe.select_dtypes(include="number").dropna()
    return [
        Correlation("pearson", get_correlation_pearson(numeric_df)),
        Correlation("spearman", get_correlation_spearman(numeric_df)),
        Correlation("kendall", get_correlation_kendall(numeric_df)),
    ]


def show_correlation(correlation, fig, position, rows, cols, cmap, title):
    ax = fig.add_subplot(rows, cols, position)
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cmap=cmap,
        ax=ax,
        cbar=True,
    )
    ax.set_title(title)
    
    
def print_correlation(correlation_list: list, idx: int):
    correlation = correlation_list[idx]
    print(correlation)


def print_correlations(correlation_list: list):
    for i in range(len(correlation_list)):
        print_correlation(correlation_list, i)
