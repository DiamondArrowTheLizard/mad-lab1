def plot_residuals(reg_model, fig, position, rows, cols):
    ax = fig.add_subplot(rows, cols, position)
    label = "Pearson residual" if reg_model.kind == "poisson" else "residual"
    ax.scatter(reg_model.fitted, reg_model.residuals, alpha=0.5, edgecolor="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"{reg_model.name} — {label} vs fitted")
    ax.set_xlabel("fitted values")
    ax.set_ylabel(label)


def plot_std_residuals_hist(reg_model, fig, position, rows, cols):
    ax = fig.add_subplot(rows, cols, position)
    ax.hist(
        reg_model.standardized_residuals,
        bins="auto",
        edgecolor="black",
        color="steelblue",
    )
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{reg_model.name} — standardized residuals")
    ax.set_xlabel("standardized residual")
    ax.set_ylabel("frequency")


def show_regression(reg_model, fig):
    fig.clear()
    plot_residuals(reg_model, fig, 1, 1, 2)
    plot_std_residuals_hist(reg_model, fig, 2, 1, 2)
    fig.tight_layout()
