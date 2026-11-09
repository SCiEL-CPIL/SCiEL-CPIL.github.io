"""
Program: GPR denoising + weighted degree-5 polynomial fit.

Strategy:
1. Estimate noise variance from local residuals.
2. Use GPR with RBF kernel to estimate E[Y|X] at well-distributed points.
3. Select length scale via marginal log-likelihood (simplified).
4. Fit a weighted degree-5 polynomial to the pseudo-labels, weighting by
   GPR predictive variance (lower variance => higher weight).
"""
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved subgrid tendency.

    Parameters:
        training_x: array of local slow-state values x
        training_subgrid_tendency: array of corresponding unresolved tendencies

    Returns:
        coeff: array of 6 coefficients in ascending powers of x.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Direct fit for very small samples
    if n < 20:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)

    # Estimate noise variance via local linear residuals
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    window = max(5, n // 20)
    noise_var_estimates = []
    step = max(1, window // 4)
    for i in range(0, n - window, step):
        xw = x_sorted[i:i + window]
        yw = y_sorted[i:i + window]
        if len(np.unique(xw)) >= 2:
            p1 = np.polyfit(xw, yw, 1)
            resid = yw - np.polyval(p1, xw)
            v = np.var(resid)
            if v > 0:
                noise_var_estimates.append(v)

    if noise_var_estimates:
        noise_var = np.median(noise_var_estimates)
    else:
        noise_var = np.var(y) * 0.1
    noise_var = max(noise_var, 1e-10)

    # Evaluation points: quantiles + a few random
    x_min, x_max = float(np.min(x)), float(np.max(x))
    n_eval = min(300, n)
    eval_points = np.quantile(x, np.linspace(0, 1, n_eval))
    n_extra = min(50, n)
    if n_extra > 0:
        extra = np.random.uniform(x_min, x_max, size=n_extra)
        eval_points = np.concatenate([eval_points, extra])
    eval_points = np.unique(eval_points)

    # Typical spacing for length-scale initialization
    dx_typical = np.median(np.diff(np.sort(x)))
    if dx_typical <= 0:
        dx_typical = (x_max - x_min) / max(n - 1, 1)

    # Try a set of length scales and pick best by marginal log-likelihood
    length_scales = np.array([0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]) * dx_typical
    length_scales = length_scales[length_scales > 0]
    if len(length_scales) == 0:
        length_scales = np.array([1.0])

    sigma_f2 = np.var(y)
    if sigma_f2 <= 0:
        sigma_f2 = 1.0

    best_ls = length_scales[0]
    best_mll = -np.inf
    best_alpha = None
    best_K_inv_y = None

    for ls in length_scales:
        try:
            diff = x[:, None] - x[None, :]
            K = np.exp(-diff ** 2 / (2.0 * ls ** 2))
            K_noisy = K + (noise_var / sigma_f2) * np.eye(n)

            # Cholesky for stability
            try:
                L = np.linalg.cholesky(K_noisy)
            except np.linalg.LinAlgError:
                # Add jitter
                L = np.linalg.cholesky(K_noisy + 1e-8 * np.eye(n))

            # Solve L L^T alpha = y
            Ly = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, Ly)

            # Marginal log-likelihood: -0.5 * y^T alpha - sum(log diag L) - n/2 log(2pi)
            mll = -0.5 * np.dot(y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

            if mll > best_mll:
                best_mll = mll
                best_ls = ls
                best_alpha = alpha.copy()
                best_K_inv_y = alpha.copy()
        except Exception:
            continue

    ls = best_ls
    alpha = best_alpha
    if alpha is None:
        # Fallback
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)

    # Predict at eval points
    K_star = np.exp(-(eval_points[:, None] - x[None, :]) ** 2 / (2.0 * ls ** 2))
    y_pseudo = K_star @ alpha

    # Predictive variance at eval points (for weighting)
    # v(x*) = sigma_f^2 - K_star K^{-1} K_star^T
    # We approximate using the Cholesky factor
    # Compute K_star @ L^{-1}
    # L from best_ls
    diff = x[:, None] - x[None, :]
    K = np.exp(-diff ** 2 / (2.0 * ls ** 2))
    K_noisy = K + (noise_var / sigma_f2) * np.eye(n)
    try:
        L = np.linalg.cholesky(K_noisy)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K_noisy + 1e-8 * np.eye(n))

    # Solve L v = K_star^T  => v = L^{-1} K_star^T
    # K_star is (n_eval, n), so K_star.T is (n, n_eval)
    try:
        V = np.linalg.solve(L, K_star.T)  # (n, n_eval)
        pred_var = sigma_f2 - np.sum(V ** 2, axis=0)  # (n_eval,)
        pred_var = np.maximum(pred_var, 1e-12)
    except Exception:
        pred_var = np.ones(len(eval_points)) * noise_var

    # Weights: inverse predictive variance (higher weight where GPR is confident)
    weights = 1.0 / pred_var
    # Normalize weights to sum to 1 for stable polyfit
    weights = weights / np.sum(weights)

    # Weighted polynomial fit
    try:
        # Use np.polynomial.polynomial.polyfit with w parameter
        coeff = np.polynomial.polynomial.polyfit(eval_points, y_pseudo, 5, w=np.sqrt(weights))
        coeff = np.asarray(coeff, dtype=np.float64)
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
        return coeff
    except Exception:
        # Fallback: unweighted polyfit on pseudo-labels
        try:
            coeff = np.polynomial.polynomial.polyfit(eval_points, y_pseudo, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        # Final fallback: direct fit on original data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            coeff = np.asarray(coeff, dtype=np.float64)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)
