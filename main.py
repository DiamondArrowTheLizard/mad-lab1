import app as a
import load_data as ld

if __name__ == "__main__":
    X, y, crosstab_list = ld.load_data()

    app = a.AutomobileApp(X, y, crosstab_list)
    app.mainloop()
