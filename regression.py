import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


class RegressionModel:
    def __init__(self, name, kind, model, formula, df, target):
        self.name = name
        self.kind = kind
        self.model = model
        self.formula = formula
        self.df = df
        self.target = target

    @property
    def fitted(self):
        return np.asarray(self.model.fittedvalues)

    @property
    def residuals(self):
        if self.kind == "poisson":
            return np.asarray(self.model.resid_pearson)
        return np.asarray(self.model.resid)

    @property
    def standardized_residuals(self):
        r = self.residuals
        return (r - r.mean()) / r.std(ddof=1)

    def equation(self):
        params = self.model.params
        terms = [f"{v:.4f}*{k}" for k, v in params.items() if k != "Intercept"]
        rhs = f"{params.get('Intercept', 0):.4f}"
        if terms:
            rhs += " + " + " + ".join(terms)
        if self.kind == "poisson":
            return f"ln(lambda) = {rhs}\nlambda    = exp({rhs})"
        return f"{self.target} = {rhs}"

    def r_squared(self):
        if self.kind == "poisson":
            return float(self.model.pseudo_rsquared(kind="mcf"))
        return float(self.model.rsquared)

    def adjusted_r_squared(self):
        if self.kind == "poisson":
            return None
        return float(self.model.rsquared_adj)

    def dispersion(self):
        if self.kind != "poisson":
            return None
        return float(self.model.pearson_chi2 / self.model.df_resid)

    def significance(self):
        if self.kind == "poisson":
            lr_stat = float(2 * (self.model.llf - self.model.llnull))
            df_model = int(len(self.model.params) - 1)
            p_value = float(stats.chi2.sf(lr_stat, df=df_model))
            return {
                "test": "LR chi2 vs null",
                "statistic": lr_stat,
                "df": df_model,
                "p_value": p_value,
            }
        return {
            "test": "F",
            "statistic": float(self.model.fvalue),
            "df_num": int(self.model.df_model),
            "df_den": int(self.model.df_resid),
            "p_value": float(self.model.f_pvalue),
        }

    def summary_text(self):
        lines = [
            f"Model:   {self.name}",
            f"Formula: {self.formula}",
            "",
            "Equation:",
            self.equation(),
            "",
        ]

        if self.kind == "poisson":
            disp = self.dispersion()
            lines.append(f"Pseudo-R2 (McFadden): {self.r_squared():.4f}")
            lines.append(f"Dispersion (Pearson chi2 / df_resid): {disp:.4f}")
            if disp > 1.5:
                lines.append("!! Possible OVERDISPERSION (dispersion >> 1).")
                lines.append(
                    "!! Consider Quasipoisson or Negative Binomial regression."
                )
            lines.append("Note: R2 and F-test are not defined for Poisson GLM.")
            lines.append(
                "Significance of the model is tested via LR chi2 vs null model."
            )
        else:
            lines.append(f"R2: {self.r_squared():.4f}")
            lines.append(f"Adjusted R2: {self.adjusted_r_squared():.4f}")

        lines.append(f"Significance: {self.significance()}")
        lines.append("")
        lines.append(str(self.model.summary()))
        return "\n".join(lines)

    def summary_table(self):
        rows = []
        rows.append(("Formula", self.formula))
        rows.append(("Equation", self.equation()))
        rows.append(("n observations", str(int(self.model.nobs))))

        if self.kind == "poisson":
            rows.append(("Pseudo-R2 (McFadden)", f"{self.r_squared():.4f}"))
            disp = self.dispersion()
            rows.append(("Dispersion (chi2/df)", f"{disp:.4f}"))
            if disp > 1.5:
                rows.append(("WARNING", "Overdispersion detected"))
        else:
            rows.append(("R2", f"{self.r_squared():.4f}"))
            rows.append(("Adjusted R2", f"{self.adjusted_r_squared():.4f}"))

        sig = self.significance()
        rows.append(("Significance test", sig["test"]))
        rows.append(("Significance stat", f"{sig['statistic']:.4f}"))
        if "df" in sig:
            rows.append(("Significance df", str(sig["df"])))
        if "df_num" in sig:
            rows.append(("Significance df (num)", str(sig["df_num"])))
            rows.append(("Significance df (den)", str(sig["df_den"])))
        rows.append(("Significance p-value", f"{sig['p_value']:.4g}"))

        rows.append(("", ""))
        pvalues = self.model.pvalues
        for name, coef in self.model.params.items():
            rows.append((f"coef: {name}", f"{coef:.4f}"))
            rows.append((f"p>|t|: {name}", f"{pvalues[name]:.4g}"))

        return rows


def _prepare(df):
    renamed = df.rename(
        columns={
            "city-mpg": "city_mpg",
            "engine-size": "engine_size",
            "fuel-type": "fuel_type",
            "body-style": "body_style",
            "drive-wheels": "drive_wheels",
        }
    )
    renamed["engine_size_sq"] = renamed["engine_size"] ** 2
    renamed["engine_size_cu"] = renamed["engine_size"] ** 3
    renamed["engine_size_sq_log"] = renamed["engine_size_sq"] * np.log(
        renamed["engine_size"]
    )

    needed = [
        "price",
        "city_mpg",
        "engine_size",
        "horsepower",
        "fuel_type",
        "body_style",
        "drive_wheels",
    ]
    return renamed.dropna(subset=needed).reset_index(drop=True)


def build_models(df):
    prepared = _prepare(df)

    linear_formula = (
        "price ~ city_mpg + engine_size + horsepower + "
        "C(fuel_type) + C(body_style) + C(drive_wheels)"
    )
    linear = smf.ols(linear_formula, data=prepared).fit()

    poisson_formula = (
        "city_mpg ~ engine_size + horsepower + "
        "C(fuel_type) + C(body_style) + C(drive_wheels)"
    )
    poisson = smf.glm(
        poisson_formula, data=prepared, family=sm.families.Poisson()
    ).fit()

    nonlinear_formula = (
        "price ~ engine_size_cu + engine_size_sq_log + city_mpg + horsepower + "
        "C(fuel_type) + C(body_style) + C(drive_wheels)"
    )
    nonlinear = smf.ols(nonlinear_formula, data=prepared).fit()

    return [
        RegressionModel("Linear", "linear", linear, linear_formula, prepared, "price"),
        RegressionModel(
            "Poisson", "poisson", poisson, poisson_formula, prepared, "city_mpg"
        ),
        RegressionModel(
            "Nonlinear", "nonlinear", nonlinear, nonlinear_formula, prepared, "price"
        ),
    ]


def compare_models(models):
    lines = ["Model comparison", "=" * 60]
    for m in models:
        sig = m.significance()
        lines.append(
            f"{m.name:10s} | R2={m.r_squared():.4f} | "
            f"{sig['test']}={sig['statistic']:.4f} | p={sig['p_value']:.4g}"
        )
    return "\n".join(lines)
