"""Rewritten program for unresolved-tendency closure task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb
from scipy.linalg import lstsq


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses regularized least-squares with cross-validation for lambda selection.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        # Not enough data for degree-5 fit; fall back to zero
        return np.zeros(6)

    # Center and scale x to improve conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Build Vandermonde matrix for polynomial basis in normalized space
    V = np.vander(x_norm, N=6, increasing=True)

    # If we have enough data, use a held-out subset for model selection
    use_cv = n > 100 and (time.time() - start_time) < (DEADLINE - 30.0)

    if use_cv:
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        x_train_norm = x_norm[train_idx]
        y_train = y[train_idx]
        x_val_norm = x_norm[val_idx]
        y_val = y[val_idx]

        V_train = np.vander(x_train_norm, N=6, increasing=True)
        V_val = np.vander(x_val_norm, N=6, increasing=True)
    else:
        x_train_norm = x_norm
        y_train = y
        V_train = V
        x_val_norm = x_norm
        y_val = y
        V_val = V

    # Try several regularization strengths
    lambdas = np.logspace(-12, 2, 15)

    best_coeff = None
    best_val_mse = np.inf

    for lam in lambdas:
        if time.time() > DEADLINE:
            break

        try:
            # Regularized least squares: solve (V^T V + lambda*I) c = V^T y
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            coeff_norm = np.linalg.solve(A, b)

            # Convert from normalized basis to raw x basis
            # P(x) = sum_{i=0}^{5} a_i * ((x - x_mean)/x_std)^i
            # = sum_{i=0}^{5} a_i / x_std^i * (x - x_mean)^i
            # = sum_{i=0}^{5} a_i / x_std^i * sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
            coeff = np.zeros(6)
            for i in range(6):
                a_i = coeff_norm[i] / (x_std ** i)
                for j in range(i + 1):
                    coeff[j] += a_i * comb(i, j) * ((-x_mean) ** (i - j))

            if not np.all(np.isfinite(coeff)):
                continue

            # Evaluate in RAW x space
            x_val_raw = x_val_norm * x_std + x_mean
            pred_val = np.zeros_like(y_val)
            for i in range(6):
                pred_val += coeff[i] * (x_val_raw ** i)
            val_mse = np.mean((pred_val - y_val) ** 2)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff = coeff.copy()

        except np.linalg.LinAlgError:
            continue

    if best_coeff is not None:
        return best_coeff

    # Fallback: unregularized least-squares fit
    try:
        coeff_norm, _, _, _ = lstsq(V_train, y_train, rcond=None)
        coeff = np.zeros(6)
        for i in range(6):
            a_i = coeff_norm[i] / (x_std ** i)
            for j in range(i + 1):
                coeff[j] += a_i * comb(i, j) * ((-x_mean) ** (i - j))
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    return np.zeros(6)
# EVOLVE-BLOCK-END
