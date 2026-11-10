"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses ridge regression with adaptive lambda selection and a time guard.
    """
    t0 = time.time()
    deadline = t0 + 120.0  # 120s wall-clock limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Basic sanity checks
    if x.size == 0 or y.size == 0:
        return np.zeros(6)

    # Ensure 1D
    x = x.ravel()
    y = y.ravel()
    n = x.size

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 6:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # y centering (optional, helps condition)
    y_mean = np.mean(y)
    y_scaled = y - y_mean

    # Build Vandermonde matrix for degree 5 in scaled x
    # We'll fit in scaled basis, then convert back to original basis
    degree = 5
    V = np.vander(x_scaled, degree=degree + 1, increasing=True)  # n x 6, columns: x^0, x^1, ..., x^5

    # Try a range of ridge parameters
    # The key insight: we want to balance fit quality and generalization
    # Use a simple grid of lambda values
    lambdas = np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])

    best_coeff = None
    best_score = np.inf

    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break

        # Ridge regression: (V^T V + lam*I) c = V^T y
        A = V.T @ V
        b = V.T @ y_scaled
        I = np.eye(degree + 1)
        # Don't regularize the intercept (constant term) as heavily
        reg = lam * I
        reg[0, 0] = 0.0  # no regularization on constant

        try:
            c_scaled = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            try:
                c_scaled = np.linalg.lstsq(A + reg, b, rcond=None)[0]
            except:
                continue

        # Evaluate training MSE (in original y scale)
        pred_scaled = V @ c_scaled
        mse = np.mean((pred_scaled - y_scaled) ** 2)
        # Use a simple validation-like score: training MSE + complexity penalty
        # But we don't have validation data, so use ridge penalty as proxy
        score = mse + lam * np.sum(c_scaled ** 2)

        if score < best_score:
            best_score = score
            best_coeff = c_scaled.copy()

    if best_coeff is None:
        # Fallback: simple OLS
        try:
            best_coeff, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
        except:
            return np.zeros(6)

    # Convert from scaled basis back to original basis
    # If x_scaled = (x - x_mean)/x_std, then x = x_scaled * x_std + x_mean
    # We need to express sum(c_i * x_scaled^i) as sum(d_j * x^j)
    # x_scaled = (x - x_mean)/x_std
    # So we need to expand each (x - x_mean)^i / x_std^i in powers of x

    d = np.zeros(degree + 1)
    for i in range(degree + 1):
        c_i = best_coeff[i]
        if abs(c_i) < 1e-15:
            continue
        # Expand (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
        for j in range(i + 1):
            from math import comb
            coeff_val = c_i / (x_std ** i) * comb(i, j) * ((-x_mean) ** (i - j))
            d[j] += coeff_val

    # Time guard: ensure we're within deadline
    if time.time() > deadline - 1.0:
        # Return quickly
        return d

    # Optional: try a few more refined lambda values around the best
    if best_score < np.inf and time.time() < deadline - 10.0:
        # Find which lambda gave best result
        for lam in lambdas:
            if time.time() > deadline - 5.0:
                break
            A = V.T @ V
            b = V.T @ y_scaled
            I = np.eye(degree + 1)
            reg = lam * I
            reg[0, 0] = 0.0
            try:
                c_scaled = np.linalg.solve(A + reg, b)
            except:
                continue
            pred_scaled = V @ c_scaled
            mse = np.mean((pred_scaled - y_scaled) ** 2)
            score = mse + lam * np.sum(c_scaled ** 2)
            if score < best_score - 1e-12:
                best_score = score
                best_coeff = c_scaled.copy()

        # Recompute d from updated best_coeff
        d = np.zeros(degree + 1)
        for i in range(degree + 1):
            c_i = best_coeff[i]
            if abs(c_i) < 1e-15:
                continue
            for j in range(i + 1):
                from math import comb
                coeff_val = c_i / (x_std ** i) * comb(i, j) * ((-x_mean) ** (i - j))
                d[j] += coeff_val

    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(d)):
        return np.zeros(6)

    return d
# EVOLVE-BLOCK-END
