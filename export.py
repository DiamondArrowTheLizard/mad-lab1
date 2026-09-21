import os


def export_all(X, y, crosstab_list, correlation_list, out_dir="data"):
    xlsx_dir = f"{out_dir}/xlsx"
    csv_dir = f"{out_dir}/csv"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(xlsx_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

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

    return out_dir