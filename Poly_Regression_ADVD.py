import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so this works in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def find_default_input_path() -> str | None:
    """Return a sensible default input path if one exists in CWD."""
    candidates = [
        "ADVD_WnWpVm_values.xlsx",
        "ADVD_WnWpVm_values.csv",
    ]
    for name in candidates:
        path = os.path.join(os.getcwd(), name)
        if os.path.exists(path):
            return path
    return None


def _rename_columns_for_convenience(df: pd.DataFrame) -> pd.DataFrame:
    """Rename commonly seen column headers to canonical names Wn, Wp, Vm.

    Handles variations like micro sign and units appearing in headers.
    """
    df = df.copy()
    lower_to_original = {c.lower(): c for c in df.columns}
    rename_pairs = [
        ("wn (µm)", "Wn"),
        ("wn (um)", "Wn"),
        ("wn", "Wn"),
        ("wp (µm)", "Wp"),
        ("wp (um)", "Wp"),
        ("wp", "Wp"),
        ("vm (v)", "Vm"),
        ("vm", "Vm"),
    ]
    rename_map: dict[str, str] = {}
    for src_lower, dst in rename_pairs:
        if src_lower in lower_to_original:
            rename_map[lower_to_original[src_lower]] = dst
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load dataframe from a CSV/XLSX file and validate required columns.

    Required columns after renaming: Wn, Wp, Vm
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path.lower())
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            df = pd.read_excel(input_path)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel requires 'openpyxl'. Install it and retry."
            ) from exc
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(
            "Unsupported file extension. Use .xlsx or .csv"
        )

    df = _rename_columns_for_convenience(df)

    required = {"Wn", "Wp", "Vm"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Input data must contain columns 'Wn', 'Wp', 'Vm'. "
            f"Found: {list(df.columns)}"
        )

    # Basic cleaning to avoid divide-by-zero and invalid entries
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Wn", "Wp", "Vm"]).copy()
    df = df[df["Wp"] != 0]
    if df.empty:
        raise ValueError("No valid rows after cleaning input data.")

    df["Ratio_WnWp"] = df["Wn"] / df["Wp"]
    return df


def generate_sample_dataframe(num_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset that roughly mimics expected behavior."""
    rng = np.random.default_rng(seed)
    wp = rng.uniform(0.5, 10.0, size=num_rows)
    wn = rng.uniform(0.5, 10.0, size=num_rows)
    ratio = wn / wp
    # Smooth nonlinear mapping to Vm around ~0.75 V for VDD=1.5 V
    vm = 0.75 + 0.35 * (ratio - 1.0) - 0.12 * (ratio - 1.0) ** 2
    vm += rng.normal(scale=0.02, size=num_rows)

    df = pd.DataFrame({"Wn": wn, "Wp": wp, "Vm": vm})
    df["Ratio_WnWp"] = df["Wn"] / df["Wp"]
    return df


def fit_and_evaluate(
    df: pd.DataFrame, degree: int, vdd: float
) -> tuple[
    LinearRegression,
    PolynomialFeatures,
    np.ndarray,
    np.ndarray,
    int,
    float,
    float,
]:
    """Fit polynomial regression and compute predictions and metrics.

    Returns: (model, poly, r_range, vm_pred, best_idx, ideal_ratio, rmse)
    """
    x = df[["Ratio_WnWp" ]].to_numpy()
    y = df["Vm"].to_numpy()

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    vm_target = vdd / 2.0
    r_range = np.linspace(df["Ratio_WnWp"].min(), df["Ratio_WnWp"].max(), 500)
    r_poly = poly.transform(r_range.reshape(-1, 1))
    vm_pred = model.predict(r_poly)

    best_idx = int(np.argmin(np.abs(vm_pred - vm_target)))
    ideal_ratio = float(r_range[best_idx])

    y_pred = model.predict(x_poly)
    rmse = float(mean_squared_error(y, y_pred, squared=False))

    return model, poly, r_range, vm_pred, best_idx, ideal_ratio, rmse


def plot_results(
    df: pd.DataFrame,
    r_range: np.ndarray,
    vm_pred: np.ndarray,
    ideal_ratio: float,
    degree: int,
    vdd: float,
    save_path: str,
    show: bool = False,
) -> str:
    """Create and save the plot, optionally showing it interactively."""
    vm_target = vdd / 2.0
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Ratio_WnWp"], df["Vm"], color="blue", alpha=0.85, label="Measured Data"
    )
    plt.plot(
        r_range,
        vm_pred,
        color="red",
        linewidth=2.0,
        label=f"Polynomial Fit (deg={degree})",
    )
    plt.axhline(vm_target, color="green", linestyle="--", label="Vm = VDD/2")
    plt.axvline(
        ideal_ratio, color="orange", linestyle="--", label=f"Ideal Ratio = {ideal_ratio:.3f}"
    )
    plt.xlabel("Wn / Wp Ratio")
    plt.ylabel("Vm (V)")
    plt.title("Polynomial Regression for Vm vs (Wn/Wp)")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    if show:
        # In headless environments this may still be ignored
        plt.show()
    plt.close()
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polynomial regression to model Vm as a function of Wn/Wp.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input .xlsx or .csv (columns: Wn, Wp, Vm).",
    )
    parser.add_argument(
        "-d", "--degree", type=int, default=3, help="Polynomial degree (default: 3)"
    )
    parser.add_argument(
        "--vdd", type=float, default=1.5, help="Supply voltage VDD (default: 1.5 V)"
    )
    parser.add_argument(
        "--save-plot",
        default=os.path.join(os.getcwd(), "poly_fit.png"),
        help="Path to save the plot image (default: ./poly_fit.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively (may not work headlessly)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Use synthetic data instead of reading a file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Source data
    if args.generate_sample:
        df = generate_sample_dataframe()
        input_desc = "synthetic sample"
    else:
        input_path = args.input or find_default_input_path()
        if input_path is None:
            warnings.warn(
                "No input file provided and no default file found. "
                "Falling back to synthetic sample data.",
                RuntimeWarning,
            )
            df = generate_sample_dataframe()
            input_desc = "synthetic sample"
        else:
            df = load_dataframe(input_path)
            input_desc = input_path

    model, poly, r_range, vm_pred, idx, ideal_ratio, rmse = fit_and_evaluate(
        df=df, degree=args.degree, vdd=args.vdd
    )

    vm_at_best = float(vm_pred[idx])
    reciprocal = float("inf") if ideal_ratio == 0 else (1.0 / ideal_ratio)

    # Output results
    print("=== Polynomial Regression Results ===")
    print(f"Data source: {input_desc}")
    print(f"Polynomial degree: {args.degree}")
    print(f"VDD: {args.vdd:.4f} V (target Vm = {args.vdd/2.0:.4f} V)")
    print(f"Ideal Wn/Wp = {ideal_ratio:.6f}")
    print(f"Ideal Wp/Wn = {reciprocal:.6f}")
    print(f"Predicted Vm at ideal ratio = {vm_at_best:.6f} V")
    print(f"Model RMSE: {rmse:.6f} V")

    save_path = plot_results(
        df=df,
        r_range=r_range,
        vm_pred=vm_pred,
        ideal_ratio=ideal_ratio,
        degree=args.degree,
        vdd=args.vdd,
        save_path=args.save_plot,
        show=args.show,
    )
    print(f"Plot written to: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
