import app as a
import crosstabs as ct
import export_csv as ec
import load_data as ld
import show_correlation as cor

if __name__ == "__main__":
    X, y, crosstab_list, correlation_list = ld.load_data()
    
    out_dir = ec.export_all(X, y, crosstab_list, correlation_list)
    print(f"Exported data to ./{out_dir}/")

    app = a.AutomobileApp(X, y, crosstab_list, correlation_list)
    app.mainloop()

    ct.print_crosstabs(crosstab_list)
    cor.print_correlations(correlation_list)
