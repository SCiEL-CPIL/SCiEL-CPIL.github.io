"""Climate dynamics: unresolved-tendency closure - degree-five polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses regularized least squares with adaptive regularization.
    """
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Build Vandermonde matrix for degree-5 polynomial
    # V[:, i] = x^i
    V = np.vander(x, N=6, increasing=True)

    # Try a range of regularization strengths via generalized cross-validation
    # or simple train/test split
    # First, do a simple polynomial fit with moderate regularization
    # to get a baseline, then refine.

    # Strategy: try several regularization alphas, pick best by internal CV
    # Use k-fold style: split into train/val
    n_val = max(int(0.2 * n), 10)
    n_train = n - n_val

    # Shuffle for validation split (use fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    V_train = np.vander(x_train, N=6, increasing=True)
    V_val = np.vander(x_val, N=6, increasing=True)

    # Scale features for better conditioning
    x_mean = x_train.mean()
    x_std = x_train.std()
    if x_std < 1e-12:
        x_std = 1.0

    # Work in scaled coordinates for stability, then transform back
    # Actually, let's just use regularization directly on the unscaled V
    # The Vandermonde can be ill-conditioned, so let's use scaled x

    V_train_s = np.vander((x_train - x_mean) / x_std, N=6, increasing=True)
    V_val_s = np.vander((x_val - x_mean) / x_std, N=6, increasing=True)

    # Try a set of alpha values (log-spaced)
    alphas = np.logspace(-8, 4, 30)

    best_alpha = 1.0
    best_val_mse = np.inf
    best_coeff_scaled = None

    for alpha in alphas:
        if time.time() - t_start > DEADLINE:
            break
        # Regularized least squares: (V^T V + alpha*I) c = V^T y
        # Add small ridge to diagonal (skip first element to not regularize constant term as heavily)
        reg = alpha * np.eye(6)
        reg[0, 0] *= 0.1  # lighter regularization on constant term

        A = V_train_s.T @ V_train_s + reg
        b = V_train_s.T @ y_train

        try:
            c_scaled = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_scaled = np.linalg.lstsq(A, b, rcond=None)[0]

        # Evaluate on validation
        pred_val = V_val_s @ c_scaled
        val_mse = np.mean((pred_val - y_val) ** 2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha
            best_coeff_scaled = c_scaled

    if best_coeff_scaled is None:
        # Fallback: simple linear fit
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Now refit on ALL data with the best alpha
    V_all_s = np.vander((x - x_mean) / x_std, N=6, increasing=True)
    reg = best_alpha * np.eye(6)
    reg[0, 0] *= 0.1
    A_all = V_all_s.T @ V_all_s + reg
    b_all = V_all_s.T @ y
    try:
        c_scaled_final = np.linalg.solve(A_all, b_all)
    except np.linalg.LinAlgError:
        c_scaled_final = np.linalg.lstsq(A_all, b_all, rcond=None)[0]

    # Transform coefficients from scaled space back to original x
    # If z = (x - mu)/sigma, then x = sigma*z + mu
    # P(x) = sum_{i=0}^{5} c_i * z^i where z = (x-mu)/sigma
    # We need P(x) = sum_{j=0}^{5} a_j * x^j
    # Use polynomial composition: substitute z = (x - mu)/sigma into the scaled polynomial
    # P(z) = c_0 + c_1*z + c_2*z^2 + ... + c_5*z^5
    # z = (x - mu)/sigma = x/sigma - mu/sigma
    # Let a = 1/sigma, b = -mu/sigma, so z = a*x + b
    # P(x) = sum_i c_i * (a*x + b)^i
    # Expand (a*x + b)^i = sum_{k=0}^{i} C(i,k) * a^k * b^{i-k} * x^k
    # So a_j = sum_{i=j}^{5} c_i * C(i,j) * a^j * b^{i-j}

    a_coef = 1.0 / x_std
    b_coef = -x_mean / x_std

    c_orig = np.zeros(6)
    for j in range(6):
        val = 0.0
        for i in range(j, 6):
            # C(i, j) * a^j * b^(i-j) * c_i
            from math import comb
            binom = comb(i, j)
            val += c_scaled_final[i] * binom * (a_coef ** j) * (b_coef ** (i - j))
        c_orig[j] = val

    # Sanity check: ensure all coefficients are finite
    if not np.all(np.isfinite(c_orig)):
        # Fallback
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    return c_orig
# EVOLVE-BLOCK-END
