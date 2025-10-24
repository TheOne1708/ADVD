from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so this works in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE with broad scikit-learn compatibility.

    Prefer sklearn's root_mean_squared_error when available; otherwise
    fall back to sqrt(mean_squared_error).
    """
    try:
        # Available in scikit-learn >= 1.4
        from sklearn.metrics import root_mean_squared_error as rmse_fn  # type: ignore

        return float(rmse_fn(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class FitResults:
    model: LinearRegression | Ridge
    poly: PolynomialFeatures
    ratio_range: np.ndarray
    vm_predicted: np.ndarray
    best_index: int
    ideal_ratio: float
    rmse: float


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


def generate_sample_dataframe(
    num_rows: int = 80, seed: int = 42, vdd: float = 1.8
) -> pd.DataFrame:
    """Generate a synthetic dataset centered around Vm ≈ VDD/2.

    The shape is a smooth nonlinear mapping of Vm vs. Wn/Wp.
    """
    rng = np.random.default_rng(seed)
    wp = rng.uniform(0.5, 10.0, size=num_rows)
    wn = rng.uniform(0.5, 10.0, size=num_rows)
    ratio = wn / wp
    vm = (vdd / 2.0) + 0.35 * (ratio - 1.0) - 0.12 * (ratio - 1.0) ** 2
    vm += rng.normal(scale=0.02, size=num_rows)

    df = pd.DataFrame({"Wn": wn, "Wp": wp, "Vm": vm})
    df["Ratio_WnWp"] = df["Wn"] / df["Wp"]
    return df


def fit_and_evaluate(
    df: pd.DataFrame,
    degree: int,
    vdd: float,
    ridge_alpha: float = 0.0,
) -> FitResults:
    """Fit polynomial regression and compute predictions and metrics.

    Uses LinearRegression by default; when ridge_alpha > 0, uses Ridge.
    """
    x = df[["Ratio_WnWp"]].to_numpy()
    y = df["Vm"].to_numpy()

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)

    if ridge_alpha and ridge_alpha > 0:
        model: LinearRegression | Ridge = Ridge(alpha=ridge_alpha)
    else:
        model = LinearRegression()
    model.fit(x_poly, y)

    vm_target = vdd / 2.0
    r_range = np.linspace(df["Ratio_WnWp"].min(), df["Ratio_WnWp"].max(), 500)
    r_poly = poly.transform(r_range.reshape(-1, 1))
    vm_pred = model.predict(r_poly)

    best_idx = int(np.argmin(np.abs(vm_pred - vm_target)))
    ideal_ratio = float(r_range[best_idx])

    y_pred = model.predict(x_poly)
    rmse = _compute_rmse(y, y_pred)

    return FitResults(
        model=model,
        poly=poly,
        ratio_range=r_range,
        vm_predicted=vm_pred,
        best_index=best_idx,
        ideal_ratio=ideal_ratio,
        rmse=rmse,
    )


def plot_results(
    df: pd.DataFrame,
    results: FitResults,
    degree: int,
    vdd: float,
    save_path: str,
    show: bool = False,
    dpi: int = 150,
) -> str:
    """Create and save the plot, optionally showing it interactively."""
    vm_target = vdd / 2.0
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Ratio_WnWp"], df["Vm"], color="blue", alpha=0.85, label="Measured Data"
    )
    plt.plot(
        results.ratio_range,
        results.vm_predicted,
        color="red",
        linewidth=2.0,
        label=f"Polynomial Fit (deg={degree})",
    )
    plt.axhline(vm_target, color="green", linestyle="--", label="Vm = VDD/2")
    plt.axvline(results.ideal_ratio, color="orange", linestyle="--", label=f"Ideal Ratio = {results.ideal_ratio:.3f}")
    plt.xlabel("Wn / Wp Ratio")
    plt.ylabel("Vm (V)")
    plt.title("Polynomial Regression for Vm vs (Wn/Wp)")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
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
        "--vdd", type=float, default=1.8, help="Supply voltage VDD (default: 1.8 V)"
    )
    parser.add_argument(
        "--save-plot",
        default=os.path.join(os.getcwd(), "poly_fit.png"),
        help="Path to save the plot image (default: ./poly_fit.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot creation and saving",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation (default: 42)",
    )
    parser.add_argument(
        "--output-json",
        help="If set, write summary results to this JSON path",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=0.0,
        help="Use Ridge regression with the given alpha (>0 to enable)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce output to warnings and errors",
    )
    return parser.parse_args()


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _summarize_results(input_desc: str, args: argparse.Namespace, results: FitResults) -> Dict[str, Any]:
    vm_at_best = float(results.vm_predicted[results.best_index])
    reciprocal = float("inf") if results.ideal_ratio == 0 else (1.0 / results.ideal_ratio)
    return {
        "data_source": input_desc,
        "degree": int(args.degree),
        "vdd": float(args.vdd),
        "target_vm": float(args.vdd / 2.0),
        "ideal_ratio_wn_wp": float(results.ideal_ratio),
        "ideal_ratio_wp_wn": float(reciprocal),
        "predicted_vm_at_ideal": float(vm_at_best),
        "rmse": float(results.rmse),
        "ridge_alpha": float(args.ridge_alpha),
    }


def main() -> int:
    args = parse_args()
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        # Sanity checks
        if args.degree < 1:
            raise ValueError("Polynomial degree must be >= 1")

        # Source data
        if args.generate_sample:
            df = generate_sample_dataframe(seed=args.seed, vdd=args.vdd)
            input_desc = "synthetic sample"
        else:
            input_path = args.input or find_default_input_path()
            if input_path is None:
                warnings.warn(
                    "No input file provided and no default file found. "
                    "Falling back to synthetic sample data.",
                    RuntimeWarning,
                )
                df = generate_sample_dataframe(seed=args.seed, vdd=args.vdd)
                input_desc = "synthetic sample"
            else:
                df = load_dataframe(input_path)
                input_desc = input_path

        results = fit_and_evaluate(
            df=df, degree=args.degree, vdd=args.vdd, ridge_alpha=args.ridge_alpha
        )

        summary = _summarize_results(input_desc=input_desc, args=args, results=results)

        logging.info("=== Polynomial Regression Results ===")
        logging.info(f"Data source: {summary['data_source']}")
        logging.info(f"Polynomial degree: {summary['degree']}")
        logging.info(
            f"VDD: {summary['vdd']:.4f} V (target Vm = {summary['target_vm']:.4f} V)"
        )
        logging.info(f"Ideal Wn/Wp = {summary['ideal_ratio_wn_wp']:.6f}")
        logging.info(f"Ideal Wp/Wn = {summary['ideal_ratio_wp_wn']:.6f}")
        logging.info(
            f"Predicted Vm at ideal ratio = {summary['predicted_vm_at_ideal']:.6f} V"
        )
        logging.info(f"Model RMSE: {summary['rmse']:.6f} V")

        if args.output_json:
            out_dir = os.path.dirname(args.output_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logging.info(f"Results JSON written to: {args.output_json}")

        if not args.no_plot:
            save_path = plot_results(
                df=df,
                results=results,
                degree=args.degree,
                vdd=args.vdd,
                save_path=args.save_plot,
                show=args.show,
            )
            logging.info(f"Plot written to: {save_path}")
        else:
            logging.debug("Skipping plot creation due to --no-plot")

        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logging.error(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
