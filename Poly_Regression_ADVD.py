import argparse
import os
import sys
import warnings
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so this works in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE with fallback for older sklearn versions."""
    # Prefer dedicated RMSE metric when available
    try:  # sklearn >= 1.4
        from sklearn.metrics import root_mean_squared_error as _sk_rmse  # type: ignore

        return float(_sk_rmse(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _augment_with_anchor(
    x: np.ndarray,
    y: np.ndarray,
    vm_target: float,
    anchor_ratio: float,
    repeats: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if repeats <= 0:
        return x, y
    anchor_x = np.full((repeats, 1), anchor_ratio, dtype=float)
    anchor_y = np.full((repeats,), vm_target, dtype=float)
    x_aug = np.concatenate([x, anchor_x], axis=0)
    y_aug = np.concatenate([y, anchor_y], axis=0)
    return x_aug, y_aug


def _build_pipeline(
    degree: int,
    regularization: Literal["none", "ridge", "lasso"],
    alpha: float,
    robust: bool,
) -> Pipeline:
    if robust:
        estimator = HuberRegressor(alpha=alpha)
    else:
        if regularization == "ridge":
            estimator = Ridge(alpha=alpha)
        elif regularization == "lasso":
            estimator = Lasso(alpha=alpha, max_iter=10000)
        else:
            estimator = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    return pipeline


def _kfold_rmse(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    regularization: Literal["none", "ridge", "lasso"],
    alpha: float,
    robust: bool,
    k_folds: int,
    random_state: int,
    vm_target: Optional[float] = None,
    anchor_ratio: float = 1.0,
    anchor_repeats: int = 0,
) -> float:
    kf = KFold(n_splits=max(2, k_folds), shuffle=True, random_state=random_state)
    rmses: list[float] = []
    for train_idx, test_idx in kf.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        if vm_target is not None and anchor_repeats > 0:
            x_train, y_train = _augment_with_anchor(
                x_train, y_train, vm_target=vm_target, anchor_ratio=anchor_ratio, repeats=anchor_repeats
            )

        pipe = _build_pipeline(
            degree=degree,
            regularization=regularization,
            alpha=alpha,
            robust=robust,
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        rmses.append(_compute_rmse(y_test, y_pred))
    return float(np.mean(rmses))


def _auto_select_degree(
    x: np.ndarray,
    y: np.ndarray,
    degree_min: int,
    degree_max: int,
    regularization: Literal["none", "ridge", "lasso"],
    alpha: float,
    robust: bool,
    k_folds: int,
    random_state: int,
    vm_target: Optional[float] = None,
    anchor_ratio: float = 1.0,
    anchor_repeats: int = 0,
) -> Tuple[int, float]:
    """Return (best_degree, cv_rmse)."""
    degrees = list(range(max(1, degree_min), max(degree_min, degree_max) + 1))
    best_deg = degrees[0]
    best_rmse = float("inf")
    for deg in degrees:
        cv_rmse = _kfold_rmse(
            x=x,
            y=y,
            degree=deg,
            regularization=regularization,
            alpha=alpha,
            robust=robust,
            k_folds=k_folds,
            random_state=random_state,
            vm_target=vm_target,
            anchor_ratio=anchor_ratio,
            anchor_repeats=anchor_repeats,
        )
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_deg = deg
    return best_deg, best_rmse


def fit_and_evaluate(
    df: pd.DataFrame,
    degree: int,
    vdd: float,
    *,
    auto_degree: bool = False,
    degree_min: int = 2,
    degree_max: int = 6,
    regularization: Literal["none", "ridge", "lasso"] = "none",
    alpha: float = 1.0,
    robust: bool = False,
    anchor_midpoint: bool = False,
    anchor_repeats: int = 0,
    k_folds: int = 5,
    random_state: int = 42,
    bootstrap: int = 0,
    ci_alpha: float = 0.05,
) -> tuple[
    Pipeline,
    PolynomialFeatures,
    np.ndarray,
    np.ndarray,
    int,
    float,
    float,
    float,
    float,
    Optional[float],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Fit a (robust/regularized) polynomial regression and compute diagnostics.

    Returns:
        (pipeline, poly, r_range, vm_pred, best_idx, ideal_ratio,
         rmse, r2, mae, cv_rmse, vm_ci_low, vm_ci_high)
    """
    x = df[["Ratio_WnWp"]].to_numpy(dtype=float)
    y = df["Vm"].to_numpy(dtype=float)

    vm_target = float(vdd) / 2.0
    anchor_ratio = 1.0

    # Auto-select degree if requested
    selected_degree = degree
    cv_rmse: Optional[float] = None
    if auto_degree:
        selected_degree, cv_rmse = _auto_select_degree(
            x=x,
            y=y,
            degree_min=degree_min,
            degree_max=degree_max,
            regularization=regularization,
            alpha=alpha,
            robust=robust,
            k_folds=k_folds,
            random_state=random_state,
            vm_target=vm_target if anchor_midpoint else None,
            anchor_ratio=anchor_ratio,
            anchor_repeats=anchor_repeats if anchor_midpoint else 0,
        )

    # Build and fit final pipeline
    pipe = _build_pipeline(
        degree=selected_degree,
        regularization=regularization,
        alpha=alpha,
        robust=robust,
    )

    x_train, y_train = x, y
    if anchor_midpoint and anchor_repeats > 0:
        x_train, y_train = _augment_with_anchor(
            x, y, vm_target=vm_target, anchor_ratio=anchor_ratio, repeats=anchor_repeats
        )

    pipe.fit(x_train, y_train)

    # Prediction grid and curve
    r_range = np.linspace(df["Ratio_WnWp"].min(), df["Ratio_WnWp"].max(), 500)
    vm_pred = pipe.predict(r_range.reshape(-1, 1))

    # Ideal ratio where Vm closest to VDD/2
    best_idx = int(np.argmin(np.abs(vm_pred - vm_target)))
    ideal_ratio = float(r_range[best_idx])

    # In-sample metrics
    y_fit = pipe.predict(x)
    rmse = _compute_rmse(y, y_fit)
    r2 = float(r2_score(y, y_fit))
    mae = float(mean_absolute_error(y, y_fit))

    # Bootstrap confidence intervals for the curve (optional)
    vm_ci_low: Optional[np.ndarray] = None
    vm_ci_high: Optional[np.ndarray] = None
    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(random_state)
        preds = np.empty((bootstrap, r_range.shape[0]), dtype=float)
        n = x.shape[0]
        for b in range(bootstrap):
            idx = rng.integers(0, n, size=n)
            x_b = x[idx]
            y_b = y[idx]
            if anchor_midpoint and anchor_repeats > 0:
                x_b, y_b = _augment_with_anchor(
                    x_b, y_b, vm_target=vm_target, anchor_ratio=anchor_ratio, repeats=anchor_repeats
                )
            p = _build_pipeline(
                degree=selected_degree,
                regularization=regularization,
                alpha=alpha,
                robust=robust,
            )
            p.fit(x_b, y_b)
            preds[b, :] = p.predict(r_range.reshape(-1, 1))
        lower_q = 100.0 * (ci_alpha / 2.0)
        upper_q = 100.0 * (1.0 - ci_alpha / 2.0)
        vm_ci_low = np.percentile(preds, lower_q, axis=0)
        vm_ci_high = np.percentile(preds, upper_q, axis=0)

    # Expose the PolynomialFeatures for compatibility
    poly: PolynomialFeatures = pipe.named_steps["poly"]

    return (
        pipe,
        poly,
        r_range,
        vm_pred,
        best_idx,
        ideal_ratio,
        rmse,
        r2,
        mae,
        cv_rmse,
        vm_ci_low,
        vm_ci_high,
    )


def plot_results(
    df: pd.DataFrame,
    r_range: np.ndarray,
    vm_pred: np.ndarray,
    ideal_ratio: float,
    degree: int,
    vdd: float,
    save_path: str,
    show: bool = False,
    vm_ci_low: Optional[np.ndarray] = None,
    vm_ci_high: Optional[np.ndarray] = None,
) -> str:
    """Create and save the plot, optionally showing it interactively."""
    vm_target = vdd / 2.0
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Ratio_WnWp"], df["Vm"], color="blue", alpha=0.85, label="Measured Data"
    )
    # Confidence band if provided
    if vm_ci_low is not None and vm_ci_high is not None:
        plt.fill_between(
            r_range,
            vm_ci_low,
            vm_ci_high,
            color="red",
            alpha=0.15,
            label="Bootstrap CI",
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
        "--auto-degree",
        action="store_true",
        help="Select degree via K-fold CV over [--degree-min, --degree-max]",
    )
    parser.add_argument(
        "--degree-min",
        type=int,
        default=2,
        help="Minimum polynomial degree to consider for auto tuning (default: 2)",
    )
    parser.add_argument(
        "--degree-max",
        type=int,
        default=6,
        help="Maximum polynomial degree to consider for auto tuning (default: 6)",
    )
    parser.add_argument(
        "--regularization",
        choices=["none", "ridge", "lasso"],
        default="none",
        help="Regularization type (default: none)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength for ridge/lasso or Huber (default: 1.0)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use HuberRegressor for robust fitting (overrides regularization choice)",
    )
    parser.add_argument(
        "--anchor-midpoint",
        action="store_true",
        help="Add repeated pseudo-observations at ratio=1 with Vm=VDD/2",
    )
    parser.add_argument(
        "--anchor-repeats",
        type=int,
        default=0,
        help="How many anchor repeats to add if --anchor-midpoint is set (default: 0)",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="K for K-fold cross validation (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV and bootstrapping (default: 42)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples for CI band (default: 0 = disabled)",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Two-sided CI alpha for bootstrap band (default: 0.05 => 95% CI)",
    )
    parser.add_argument(
        "--report-cv",
        action="store_true",
        help="Always report CV RMSE for the selected degree",
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

    (
        pipe,
        poly,
        r_range,
        vm_pred,
        idx,
        ideal_ratio,
        rmse,
        r2,
        mae,
        cv_rmse,
        vm_ci_low,
        vm_ci_high,
    ) = fit_and_evaluate(
        df=df,
        degree=args.degree,
        vdd=args.vdd,
        auto_degree=args.auto_degree,
        degree_min=args.degree_min,
        degree_max=args.degree_max,
        regularization=args.regularization,
        alpha=args.alpha,
        robust=args.robust,
        anchor_midpoint=args.anchor_midpoint,
        anchor_repeats=args.anchor_repeats,
        k_folds=args.kfolds,
        random_state=args.random_state,
        bootstrap=args.bootstrap,
        ci_alpha=args.ci_alpha,
    )

    # Derived values
    vm_at_best = float(vm_pred[idx])
    reciprocal = float("inf") if ideal_ratio == 0 else (1.0 / ideal_ratio)
    selected_degree = int(poly.degree)  # type: ignore[attr-defined]

    # Output results
    print("=== Polynomial Regression Results ===")
    print(f"Data source: {input_desc}")
    print(
        f"Polynomial degree: {selected_degree}" + (" (auto-selected)" if args.auto_degree else "")
    )
    reg_desc = "Huber (robust)" if args.robust else (
        "Ridge" if args.regularization == "ridge" else ("Lasso" if args.regularization == "lasso" else "LinearRegression")
    )
    if args.robust or args.regularization in {"ridge", "lasso"}:
        print(f"Estimator: {reg_desc} (alpha={args.alpha})")
    else:
        print(f"Estimator: {reg_desc}")
    if args.anchor_midpoint and args.anchor_repeats > 0:
        print(f"Anchor: ratio=1.0 -> Vm=VDD/2 repeated {args.anchor_repeats}x")
    print(f"VDD: {args.vdd:.4f} V (target Vm = {args.vdd/2.0:.4f} V)")
    print(f"Ideal Wn/Wp = {ideal_ratio:.6f}")
    print(f"Ideal Wp/Wn = {reciprocal:.6f}")
    print(f"Predicted Vm at ideal ratio = {vm_at_best:.6f} V")
    print(f"Train RMSE: {rmse:.6f} V | R2: {r2:.4f} | MAE: {mae:.6f} V")
    if args.auto_degree or args.report_cv:
        if cv_rmse is None:
            # Ensure CV reported for the selected degree
            x = df[["Ratio_WnWp"]].to_numpy(dtype=float)
            y = df["Vm"].to_numpy(dtype=float)
            cv_rmse = _kfold_rmse(
                x=x,
                y=y,
                degree=selected_degree,
                regularization=args.regularization,
                alpha=args.alpha,
                robust=args.robust,
                k_folds=args.kfolds,
                random_state=args.random_state,
                vm_target=(args.vdd/2.0) if args.anchor_midpoint else None,
                anchor_ratio=1.0,
                anchor_repeats=args.anchor_repeats if args.anchor_midpoint else 0,
            )
        print(f"CV RMSE ({args.kfolds}-fold): {cv_rmse:.6f} V")

    save_path = plot_results(
        df=df,
        r_range=r_range,
        vm_pred=vm_pred,
        ideal_ratio=ideal_ratio,
        degree=selected_degree,
        vdd=args.vdd,
        save_path=args.save_plot,
        show=args.show,
        vm_ci_low=vm_ci_low,
        vm_ci_high=vm_ci_high,
    )
    print(f"Plot written to: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
