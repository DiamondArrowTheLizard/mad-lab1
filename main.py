import app as a
import crosstabs as ct
import export as e
import load_data as ld
import regression as reg
import show_correlation as cor

if __name__ == "__main__":
    X, y, crosstab_list, correlation_list, describe_list, regression_models = (
        ld.load_data()
    )

    out_dir = e.export_all(
        X, y, crosstab_list, correlation_list, describe_list, regression_models
    )
    print(f"Exported data to ./{out_dir}/")
    print(reg.compare_models(regression_models))

    application = a.AutomobileApp(
        X, y, crosstab_list, correlation_list, describe_list, regression_models
    )
    application.mainloop()

    ct.print_crosstabs(crosstab_list)
    cor.print_correlations(correlation_list)
