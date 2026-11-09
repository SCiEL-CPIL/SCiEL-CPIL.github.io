"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # If very few points, fall back to lower degree
    if n < 6:
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6, dtype=np.float64)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Scale x to [-1, 1] range for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)

    x_center = (x_min + x_max) / 2.0
    x_half_range = x_range / 2.0
    x_scaled = (x - x_center) / x_half_range

    # Build Vandermonde matrix for degree 5
    V = np.vander(x_scaled, N=6, increasing=True)

    # Compute V^T V and V^T y once
    VtV = V.T @ V
    Vty = V.T @ y

    # Try multiple regularization strengths with a simple holdout validation
    best_coeff = np.zeros(6, dtype=np.float64)
    best_val_mse = np.inf

    # Use a fraction of data for validation
    val_fraction = 0.2
    val_size = max(100, int(n * val_fraction))
    val_size = min(val_size, n // 2)

    # Shuffle indices for validation split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    V_train = V[train_idx]
    y_train = y[train_idx]
    V_val = V[val_idx]
    y_val = y[val_idx]

    # Precompute V_train^T V_train and V_train^T y_train
    VtV_train = V_train.T @ V_train
    Vty_train = V_train.T @ y_train

    # Regularization strengths: relative to data scale
    # Use a geometric sequence centered around a reasonable default
    trace_VtV = np.trace(VtV_train)
    if trace_VtV < 1e-12:
        trace_VtV = 1.0

    # Lambda values as fraction of trace
    lambda_fracs = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    for frac in lambda_fracs:
        if time.time() - t_start > deadline - 5.0:
            break
        try:
            lam = frac * trace_VtV / 6.0
            A = VtV_train + lam * np.eye(6)
            coeff_scaled = np.linalg.solve(A, Vty_train)

            # Compute validation MSE
            pred_val = V_val @ coeff_scaled
            val_mse = np.mean((pred_val - y_val) ** 2)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff = coeff_scaled
        except Exception:
            continue

    # If all regularization attempts failed, fall back to least squares
    if np.all(best_coeff == 0) or not np.isfinite(best_val_mse):
        try:
            best_coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        except Exception:
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 1)
                full = np.zeros(6, dtype=np.float64)
                full[:2] = coeff
                return full
            except Exception:
                return np.zeros(6, dtype=np.float64)

    # Convert coefficients from x_scaled to original x scale
    # x_scaled = (x - c) / s where c = x_center, s = x_half_range
    c = x_center
    s = x_half_range

    # p(x_scaled) = sum_{i=0}^{5} a_i * x_scaled^i
    # x_scaled = (x - c) / s
    # p = sum a_i * ((x-c)/s)^i = sum a_i * s^{-i} * (x-c)^i
    # Expand (x-c)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-c)^(i-j)

    coeff_orig = np.zeros(6, dtype=np.float64)
    for i in range(6):
        a_i = best_coeff[i]
        if abs(a_i) < 1e-15:
            continue
        for j in range(i + 1):
            binom_coeff = comb(i, j)
            power_c = (-c) ** (i - j)
            coeff_orig[j] += a_i * (binom_coeff * power_c) / (s ** i)

    # Verify finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to simple fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            full = np.zeros(6, dtype=np.float64)
            full[:4] = coeff
            return full
        except Exception:
            return np.zeros(6, dtype=np.float64)

    return coeff_orig.astype(np.float64)
# EVOLVE-BLOCK-END
