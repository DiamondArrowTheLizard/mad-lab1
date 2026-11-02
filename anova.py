import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm


class ANOVAResult:
    def __init__(self, name, kind, summary, data=None):
        self.name = name
        self.kind = kind
        self.summary = summary
        self.data = data or {}

    def summary_text(self):
        return self.summary

    def __str__(self):
        return f"ANOVA ({self.name})\n{self.summary}"


def _prepare(df):
    renamed = df.rename(
        columns={
            "body-style": "body_style",
            "drive-wheels": "drive_wheels",
            "fuel-type": "fuel_type",
        }
    )
    needed = ["price", "body_style", "drive_wheels"]
    needed = [c for c in needed if c in renamed.columns]
    return renamed.dropna(subset=needed).reset_index(drop=True)


def _safe_shapiro(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3 or np.nanstd(values) == 0:
        return None
    try:
        return stats.shapiro(values)
    except Exception:  # noqa: BLE001
        return None


def _shapiro_line(label, values):
    result = _safe_shapiro(values)
    if result is None:
        return f"Shapiro-Wilk {label}: not available"
    return f"Shapiro-Wilk {label}: W={result.statistic:.4f}, p={result.pvalue:.4g}"


def _descriptive_by_group(df, dep, factor):
    groups = {}
    lines = [
        "Descriptive statistics by group",
        f"{'group':<20}{'n':>6}{'mean':>12}{'std':>12}{'median':>12}{'min':>12}{'max':>12}",
    ]
    levels = sorted(df[factor].dropna().unique(), key=lambda x: str(x))
    for level in levels:
        values = df.loc[df[factor] == level, dep].dropna()
        groups[str(level)] = values
        if len(values) == 0:
            continue
        std = values.std(ddof=1) if len(values) > 1 else float("nan")
        lines.append(
            f"{level!s:<20}{len(values):>6}{values.mean():>12.4f}{std:>12.4f}"
            f"{values.median():>12.4f}{values.min():>12.4f}{values.max():>12.4f}"
        )
    return groups, lines


def one_way_anova(df, dep, factor):
    data = df.dropna(subset=[dep, factor]).copy()
    groups, desc_lines = _descriptive_by_group(data, dep, factor)

    lines = [
        f"One-way ANOVA: {dep} ~ {factor}",
        "",
        "H0: all group means are equal",
        "H1: at least one group mean differs",
        "",
    ]
    lines.extend(desc_lines)
    lines.append("")
    lines.append(_shapiro_line("dependent variable", data[dep]))

    arrays = []
    for level, values in groups.items():
        lines.append(_shapiro_line(f"group {level}", values))
        if len(values) >= 2:
            arrays.append(values.values)

    if len(arrays) >= 2:
        f_stat, f_p = stats.f_oneway(*arrays)
        levene = stats.levene(*arrays)
        kruskal = stats.kruskal(*arrays)

        lines.append("")
        lines.append(f"Fisher ANOVA: F={f_stat:.4f}, p={f_p:.4g}")
        lines.append(f"Levene test: W={levene.statistic:.4f}, p={levene.pvalue:.4g}")
        lines.append(
            f"Kruskal-Wallis: H={kruskal.statistic:.4f}, p={kruskal.pvalue:.4g}"
        )
        lines.append(
            f"ANOVA conclusion: {'reject H0' if f_p < 0.05 else 'do not reject H0'} at alpha=0.05"
        )
        lines.append(
            f"Kruskal-Wallis conclusion: {'reject H0' if kruskal.pvalue < 0.05 else 'do not reject H0'} at alpha=0.05"
        )
        lines.append(
            "Use parametric ANOVA when normality and homogeneity assumptions are acceptable; otherwise prefer Kruskal-Wallis."
        )

        try:
            model = smf.ols(f"{dep} ~ C({factor})", data=data).fit()
            table = anova_lm(model, typ=2)
            lines.append("")
            lines.append("ANOVA table")
            lines.append(table.to_string())
        except Exception as exc:  # noqa: BLE001
            table = pd.DataFrame()
            lines.append(f"ANOVA table error: {exc}")
    else:
        table = pd.DataFrame()
        lines.append("Not enough groups with at least two observations for ANOVA")

    group_describe = {level: values.describe() for level, values in groups.items()}
    return ANOVAResult(
        f"One-way ANOVA: {dep} ~ {factor}",
        "one_way",
        "\n".join(lines),
        {"groups": group_describe, "anova_table": table},
    )


def two_way_anova(df, dep, factor1, factor2):
    data = df.dropna(subset=[dep, factor1, factor2]).copy()
    lines = [
        f"Two-way ANOVA: {dep} ~ {factor1} * {factor2}",
        "",
        "H0: factor effects and interaction are absent",
        "H1: at least one effect or interaction is present",
        "",
    ]

    try:
        formula = f"{dep} ~ C({factor1}) * C({factor2})"
        model = smf.ols(formula, data=data).fit()
        table = anova_lm(model, typ=2)
        lines.append(table.to_string())
        lines.append("")
        lines.append(f"R2={model.rsquared:.4f}, adjusted R2={model.rsquared_adj:.4f}")
    except Exception as exc:  # noqa: BLE001
        table = pd.DataFrame()
        lines.append(f"Two-way ANOVA error: {exc}")

    return ANOVAResult(
        f"Two-way ANOVA: {dep} ~ {factor1} * {factor2}",
        "two_way",
        "\n".join(lines),
        {"anova_table": table},
    )


def generate_anova(dataframe):
    prepared = _prepare(dataframe)
    results = []

    for factor in ["body_style", "drive_wheels"]:
        if factor in prepared.columns and prepared[factor].nunique() > 1:
            results.append(one_way_anova(prepared, "price", factor))

    if (
        "body_style" in prepared.columns
        and "drive_wheels" in prepared.columns
        and prepared["body_style"].nunique() > 1
        and prepared["drive_wheels"].nunique() > 1
    ):
        results.append(two_way_anova(prepared, "price", "body_style", "drive_wheels"))

    return results
