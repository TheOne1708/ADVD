from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Headless-safe plotting; only used if --save-plot is provided
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


DEFAULT_INPUT_PATH = r"C:\Users\Aaditya Rajput\Downloads\Inv_WnWpVm_values.xlsx"
DEFAULT_TARGET_VM = 0.9
DEFAULT_DEGREE = 3


@dataclass
class FitArtifacts:
    model: LinearRegression
    poly: PolynomialFeatures


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        # scikit-learn >= 1.4
        from sklearn.metrics import root_mean_squared_error as rmse_fn  # type: ignore
        return float(rmse_fn(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rename_columns_for_convenience(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_to_original = {c.lower(): c for c in df.columns}
    rename_pairs = [
        ("wn (µm)", "Wn"), ("wn (um)", "Wn"), ("wn", "Wn"),
        ("wp (µm)", "Wp"), ("wp (um)", "Wp"), ("wp", "Wp"),
        ("vm (v)", "Vm"), ("vm", "Vm"),
    ]
    rename_map: Dict[str, str] = {}
    for src_lower, dst in rename_pairs:
        if src_lower in lower_to_original:
            rename_map[lower_to_original[src_lower]] = dst
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_dataframe(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path.lower())
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file extension. Use .xlsx or .csv")

    df = _rename_columns_for_convenience(df)

    required = {"Wn", "Wp", "Vm"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Input data must contain columns 'Wn', 'Wp', 'Vm'. "
            f"Found: {list(df.columns)}"
        )

    # Clean and derive ratio
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Wn", "Wp", "Vm"]).copy()
    df = df[df["Wp"] != 0]
    if df.empty:
        raise ValueError("No valid rows after cleaning input data.")

    df["Ratio_WnWp"] = df["Wn"] / df["Wp"]
    return df


def generate_sample_dataframe(num_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wp = rng.uniform(0.5, 10.0, size=num_rows)
    wn = rng.uniform(0.5, 10.0, size=num_rows)
    ratio = wn / wp
    # Nonlinear mapping around ~0.9 V
    vm = 0.9 + 0.4 * (ratio - 1.0) - 0.18 * (ratio - 1.0) ** 2
    vm += rng.normal(scale=0.02, size=num_rows)

    df = pd.DataFrame({"Wn": wn, "Wp": wp, "Vm": vm})
    df["Ratio_WnWp"] = df["Wn"] / df["Wp"]
    return df


def fit_polynomial(df: pd.DataFrame, degree: int) -> Tuple[FitArtifacts, float]:
    x = df[["Ratio_WnWp"]].to_numpy()
    y = df["Vm"].to_numpy()

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    y_pred = model.predict(x_poly)
    rmse = _compute_rmse(y, y_pred)

    return FitArtifacts(model=model, poly=poly), rmse


def find_ratio_for_target_vm(
    artifacts: FitArtifacts,
    df: pd.DataFrame,
    target_vm: float,
    num_points: int = 1000,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    ratio_min = float(df["Ratio_WnWp"].min())
    ratio_max = float(df["Ratio_WnWp"].max())
    ratio_range = np.linspace(ratio_min, ratio_max, num_points)

    r_poly = artifacts.poly.transform(ratio_range.reshape(-1, 1))
    vm_pred = artifacts.model.predict(r_poly)

    best_idx = int(np.argmin(np.abs(vm_pred - target_vm)))
    best_ratio_wn_wp = float(ratio_range[best_idx])

    return best_ratio_wn_wp, float(vm_pred[best_idx]), ratio_range, vm_pred


def maybe_save_plot(
    df: pd.DataFrame,
    ratio_range: np.ndarray,
    vm_pred: np.ndarray,
    best_ratio: float,
    target_vm: float,
    save_path: str | None,
    degree: int,
) -> None:
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Ratio_WnWp"], df["Vm"], color="blue", alpha=0.85, label="Data")
    plt.plot(ratio_range, vm_pred, color="red", linewidth=2.0, label=f"Poly fit (deg={degree})")
    plt.axhline(target_vm, color="green", linestyle="--", label=f"Target Vm = {target_vm:.3f} V")
    plt.axvline(best_ratio, color="orange", linestyle="--", label=f"Ideal Wn/Wp = {best_ratio:.3f}")
    plt.xlabel("Wn / Wp Ratio")
    plt.ylabel("Vm (V)")
    plt.title("Polynomial Regression: Vm vs (Wn/Wp)")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find ideal Wn/Wp and Wp/Wn ratios for a target Vm using polynomial regression.",
    )
    p.add_argument("-i", "--input", default=DEFAULT_INPUT_PATH, help="Path to input .xlsx/.csv with columns Wn, Wp, Vm")
    p.add_argument("-d", "--degree", type=int, default=DEFAULT_DEGREE, help="Polynomial degree (default: 3)")
    p.add_argument("--target-vm", type=float, default=DEFAULT_TARGET_VM, help="Target Vm in volts (default: 0.9)")
    p.add_argument("--save-plot", default=None, help="Optional path to save plot image")
    p.add_argument("--output-json", default=None, help="Optional path to write summary JSON")
    p.add_argument("--generate-sample", action="store_true", help="Use synthetic data instead of reading a file")
    p.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data (when --generate-sample)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load data
    if args.generate_sample:
        df = generate_sample_dataframe(seed=args.seed)
        input_desc = "synthetic sample"
    else:
        df = load_dataframe(args.input)
        input_desc = args.input

    # Fit model
    artifacts, rmse = fit_polynomial(df=df, degree=args.degree)

    # Search for best ratio achieving target Vm
    best_ratio_wn_wp, vm_at_best, ratio_range, vm_pred = find_ratio_for_target_vm(
        artifacts=artifacts, df=df, target_vm=args.target_vm
    )
    best_ratio_wp_wn = float("inf") if best_ratio_wn_wp == 0 else (1.0 / best_ratio_wn_wp)

    # Report
    summary: Dict[str, Any] = {
        "data_source": input_desc,
        "degree": int(args.degree),
        "target_vm": float(args.target_vm),
        "ideal_ratio_wn_wp": float(best_ratio_wn_wp),
        "ideal_ratio_wp_wn": float(best_ratio_wp_wn),
        "predicted_vm_at_ideal": float(vm_at_best),
        "rmse": float(rmse),
    }

    print(json.dumps(summary, indent=2))

    # Optional outputs
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_plot:
        maybe_save_plot(
            df=df,
            ratio_range=ratio_range,
            vm_pred=vm_pred,
            best_ratio=best_ratio_wn_wp,
            target_vm=args.target_vm,
            save_path=args.save_plot,
            degree=args.degree,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
