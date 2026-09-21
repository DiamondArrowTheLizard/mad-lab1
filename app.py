import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import show_correlation as sc
import show_histogram as sh


class AutomobileApp(tk.Tk):
    def __init__(self, X, y, crosstab_list, correlation_list):
        super().__init__()
        self.title("Automobile Dataset")
        self.geometry("1100x750")

        self.X = X
        self.y = y
        self.crosstab_list = crosstab_list
        self.correlation_list = correlation_list

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._build_histogram_tab()
        self._build_crosstab_tab()
        self._build_correlation_tab()

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------
    def _build_histogram_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Histograms")

        fig = Figure(figsize=(9, 6), dpi=100)

        sh.show_histogram(
            fig,
            1,
            2,
            2,
            self.X["body-style"],
            "blue",
            "black",
            "body style",
            "types",
            "amount",
        )
        sh.show_histogram(
            fig,
            2,
            2,
            2,
            self.X["fuel-type"],
            "red",
            "black",
            "fuel type",
            "types",
            "amount",
        )
        sh.show_histogram(
            fig,
            3,
            2,
            2,
            self.X["num-of-cylinders"],
            "orange",
            "black",
            "num of cylinders",
            "types",
            "amount",
        )
        sh.show_histogram(
            fig,
            4,
            2,
            2,
            self.X["fuel-system"],
            "purple",
            "black",
            "fuel system",
            "types",
            "amount",
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Crosstabs
    # ------------------------------------------------------------------
    def _build_crosstab_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Crosstabs")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Select crosstab:").pack(side="left")

        self.crosstab_names = [f"{c.var1} vs {c.var2}" for c in self.crosstab_list]
        self.combo = ttk.Combobox(top, values=self.crosstab_names, state="readonly")
        self.combo.current(0)
        self.combo.pack(side="left", padx=6, fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._on_crosstab_select)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.tree = ttk.Treeview(tree_frame, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._show_crosstab(0)

    def _on_crosstab_select(self, _event=None):
        self._show_crosstab(self.combo.current())

    def _show_crosstab(self, idx: int):
        df = self.crosstab_list[idx].data

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()

        columns = ["index"] + [str(c) for c in df.columns]
        self.tree["columns"] = columns
        self.tree["show"] = "headings"

        self.tree.heading("index", text=df.index.name or "")
        self.tree.column("index", width=140, anchor="w")

        for col in df.columns:
            self.tree.heading(str(col), text=str(col))
            self.tree.column(str(col), width=90, anchor="center")

        for row_idx, row in df.iterrows():
            self.tree.insert("", "end", values=[row_idx] + list(row))

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------
    def _build_correlation_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Correlations")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Method:").pack(side="left")

        self.correlation_names = [c.name for c in self.correlation_list]
        self.corr_combo = ttk.Combobox(
            top, values=self.correlation_names, state="readonly"
        )
        self.corr_combo.current(0)
        self.corr_combo.pack(side="left", padx=6)
        self.corr_combo.bind("<<ComboboxSelected>>", self._on_correlation_select)

        self.corr_fig = Figure(figsize=(10, 8), dpi=100)
        self.corr_canvas = FigureCanvasTkAgg(self.corr_fig, master=frame)
        self.corr_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._show_correlation(0)

    def _on_correlation_select(self, _event=None):
        self._show_correlation(self.corr_combo.current())

    def _show_correlation(self, idx: int):
        self.corr_fig.clear()
        corr = self.correlation_list[idx]
        sc.show_correlation(
            corr.data,
            self.corr_fig,
            1,
            1,
            1,
            cmap="coolwarm",
            title=f"{corr.name.capitalize()} correlation",
        )
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
