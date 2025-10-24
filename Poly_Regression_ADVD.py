import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Load data ===
file_path = r"C:\Users\Aaditya Rajput\Downloads\ADVD_WnWpVm_values.xlsx"
df = pd.read_excel(file_path)

# Rename for convenience
df.rename(columns={"Wn (µm)": "Wn", "Wp (µm)": "Wp", "Vm (V)": "Vm"}, inplace=True)

# Compute ratio Wn/Wp
df["Ratio_WnWp"] = df["Wn"] / df["Wp"]

# === Polynomial regression: Vm = f(Wn/Wp) ===
X = df[["Ratio_WnWp"]]
y = df["Vm"]

# Polynomial degree
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# === Target Vm ===
VDD = 1.5  # change this if your circuit uses a different supply
Vm_target = VDD / 2

# Predict over smooth range of ratios
r_range = np.linspace(df["Ratio_WnWp"].min(), df["Ratio_WnWp"].max(), 500)
r_poly = poly.transform(r_range.reshape(-1, 1))
Vm_pred = model.predict(r_poly)

# Find ratio giving Vm closest to VDD/2
idx = np.argmin(np.abs(Vm_pred - Vm_target))
ideal_ratio = r_range[idx]

# === Output results ===
print("=== Ideal Ratio for Vm = VDD/2 ===")
print(f"Ideal Wn/Wp = {ideal_ratio:.4f}")
print(f"Ideal Wp/Wn = {1/ideal_ratio:.4f}")
print(f"Predicted Vm at that ratio = {Vm_pred[idx]:.4f} V")

# Evaluate fit
y_pred = model.predict(X_poly)
rmse = mean_squared_error(y, y_pred, squared=False)
print(f"\nModel RMSE: {rmse:.6f} V")

# === Plot ===
plt.figure(figsize=(7,5))
plt.scatter(df["Ratio_WnWp"], df["Vm"], color="blue", label="Measured Data")
plt.plot(r_range, Vm_pred, color="red", label=f"Polynomial Fit (deg={degree})")
plt.axhline(Vm_target, color="green", linestyle="--", label="Vm = VDD/2")
plt.axvline(ideal_ratio, color="orange", linestyle="--", label=f"Ideal Ratio = {ideal_ratio:.3f}")
plt.xlabel("Wn / Wp Ratio")
plt.ylabel("Vm (V)")
plt.title("Polynomial Regression for Vm vs (Wn/Wp)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
