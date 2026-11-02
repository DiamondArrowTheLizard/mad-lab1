import os

import pandas as pd


def _slug(value):
    return "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)
    ).strip("_")


def export_all(
    X,
    y,
    crosstab_list,
    correlation_list,
    describe_list,
    regression_models,
    anova_results,
    out_dir="data",
):
    xlsx_dir = f"{out_dir}/xlsx"
    csv_dir = f"{out_dir}/csv"
    reports_dir = f"{out_dir}/reports"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(xlsx_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "targets.csv"), index=False)

    for crosstab in crosstab_list:
        filename = f"crosstab_{crosstab.var1}_vs_{crosstab.var2}.csv"
        crosstab.data.to_csv(os.path.join(csv_dir, filename))

        filename = f"crosstab_{crosstab.var1}_vs_{crosstab.var2}.xlsx"
        crosstab.data.to_excel(os.path.join(xlsx_dir, filename))

    for correlation in correlation_list:
        filename = f"correlation_{correlation.name}.csv"
        correlation.data.to_csv(os.path.join(csv_dir, filename))

        filename = f"correlation_{correlation.name}.xlsx"
        correlation.data.to_excel(os.path.join(xlsx_dir, filename))

    for describe in describe_list:
        df = describe.data.to_frame(name=describe.name)

        filename = f"describe_{describe.name}.csv"
        df.to_csv(os.path.join(csv_dir, filename))

        filename = f"describe_{describe.name}.xlsx"
        df.to_excel(os.path.join(xlsx_dir, filename))

    combined = []
    for reg_model in regression_models:
        combined.append("=" * 80)
        combined.append(reg_model.summary_text())
        combined.append("")

        filename = f"regression_{reg_model.name.lower()}_summary.txt"
        with open(os.path.join(reports_dir, filename), "w", encoding="utf-8") as f:
            f.write(reg_model.summary_text())

    for anova in anova_results:
        combined.append("=" * 80)
        combined.append(anova.summary_text())
        combined.append("")

        filename = f"anova_{_slug(anova.name)}.txt"
        with open(os.path.join(reports_dir, filename), "w", encoding="utf-8") as f:
            f.write(anova.summary_text())

        for key, value in anova.data.items():
            if isinstance(value, pd.DataFrame):
                frame = value
            elif isinstance(value, dict):
                frame = pd.DataFrame(value)
            else:
                continue

            if frame.empty:
                continue

            base = f"anova_{_slug(anova.name)}_{_slug(key)}"
            frame.to_csv(os.path.join(csv_dir, base + ".csv"))
            frame.to_excel(os.path.join(xlsx_dir, base + ".xlsx"))

    with open(
        os.path.join(reports_dir, "regression_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(combined))

    return out_dir
