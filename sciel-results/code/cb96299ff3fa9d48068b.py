"""Rewritten program: robust degree-5 polynomial closure with regularization and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any non-finite entries
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Normalize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Build Vandermonde matrix for degree 5
    V = np.vander(x_norm, N=6, increasing=True)

    # Ridge regression with small regularization for stability
    # Use multiple lambda values and pick the one that gives best condition
    best_coeff = None
    best_score = np.inf

    for lam in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        if time.time() > deadline - 10.0:
            break
        try:
            A = V.T @ V + lam * np.eye(6)
            b = V.T @ y_norm
            coeff_norm = np.linalg.solve(A, b)
            # Transform back to original scale
            # y = y_mean + y_std * sum(c_i * (x_norm)^i)
            # x_norm = (x - x_mean) / x_std
            # So we need to express in terms of x
            # Actually, let's just use the normalized coefficients directly
            # and transform back
            # y_pred = y_mean + y_std * P(x_norm)
            # where P is the polynomial in x_norm
            # We need coefficients in x: P(x_norm) = P((x-x_mean)/x_std)
            # This is a polynomial transformation
            coeff_orig = _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std)
            # Compute training MSE
            pred = np.polynomial.polynomial.polyval(x, coeff_orig)
            mse = np.mean((pred - y) ** 2)
            if mse < best_score:
                best_score = mse
                best_coeff = coeff_orig
        except Exception:
            continue

    if best_coeff is None:
        # Fallback: least squares with SVD
        try:
            V_full = np.vander(x, N=6, increasing=True)
            coeff, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
            best_coeff = coeff
        except Exception:
            best_coeff = np.zeros(6)

    # Optional: local polish using scipy minimize on the original scale
    if time.time() < deadline - 15.0:
        try:
            def objective(coeffs):
                pred = np.polynomial.polynomial.polyval(x, coeffs)
                return np.mean((pred - y) ** 2)

            # Use a few random restarts
            best_polish = best_coeff.copy()
            best_obj = objective(best_coeff)

            n_restarts = 3
            rng = np.random.RandomState(42)
            for r in range(n_restarts):
                if time.time() > deadline - 10.0:
                    break
                perturbation = rng.randn(6) * 0.01 * np.max(np.abs(best_coeff)) if np.max(np.abs(best_coeff)) > 0 else rng.randn(6) * 0.1
                x0 = best_coeff + perturbation
                res = minimize(objective, x0, method='L-BFGS-B', maxiter=100)
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_polish = res.x.copy()

            best_coeff = best_polish
        except Exception:
            pass

    # Ensure finite output
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)

    return best_coeff


def _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    = y_mean + y_std * sum_{i=0}^{5} c_i * (1/x_std)^i * (x - x_mean)^i
    Expand (x - x_mean)^i using binomial theorem.
    """
    # coeff_norm[i] is the coefficient of (x_norm)^i
    # y = y_mean + y_std * sum_i c_i * (x_norm)^i
    # x_norm = (x - x_mean) / x_std
    # So y = y_mean + y_std * sum_i c_i * (x - x_mean)^i / x_std^i
    # We need to express sum_i c_i * (x - x_mean)^i / x_std^i as sum_j d_j * x^j
    # (x - x_mean)^i = sum_{k=0}^{i} binom(i,k) * x^k * (-x_mean)^(i-k)
    # So the coefficient of x^j in the expansion is:
    # d_j = y_std * sum_{i=j}^{5} c_i / x_std^i * binom(i, j) * (-x_mean)^(i-j)
    d = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_val = _binom(i, j)
            term = coeff_norm[i] / (x_std ** i) * binom_val * ((-x_mean) ** (i - j))
            s += term
        d[j] = y_std * s
    d[0] += y_mean
    return d


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
# EVOLVE-BLOCK-END
