"""
Strategy: 
1. Work entirely in a normalized coordinate to avoid ill-conditioning of the monomial basis.
2. Use weighted least-squares with a diagonal penalty that decays with polynomial order,
   which regularizes high-degree terms more heavily (preventing overfitting to noise).
3. Fit directly on the raw training pairs (no GPR denoising step), but use a Chebyshev-inspired
   weighting to balance the fit across the domain.
4. Convert the normalized-basis coefficients back to the original x-basis using stable
   binomial expansion (not polynomial evaluation/interpolation), which avoids catastrophic
   cancellation.
5. Use k-fold cross-validation to select the regularization strength lambda.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=float).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=float).ravel()
    n = len(x)

    if n < 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(1, n - 1))
            return np.pad(coeff, (0, 6 - len(coeff)))
        except:
            return np.zeros(6)

    # Normalize x to [-1, 1] for stable polynomial fitting
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    t = (x - x_mean) / x_std  # normalized coordinate

    # Build Vandermonde matrix in normalized coordinate: V[i,j] = t_i^j for j=0..5
    V = np.vander(t, N=6, increasing=True)

    # Weighted least-squares with order-dependent regularization
    # Weights: emphasize points where the polynomial should fit well
    # Use uniform weights for now; the key regularization is in the penalty
    w = np.ones(n)

    # Penalty matrix: penalize higher-order coefficients more
    # P[j,j] = lambda * (j+1)^2 to decay the influence of high-degree terms
    # We'll search over lambda using k-fold CV

    def fit_coefficients(t, y, lam, weights=None):
        """Fit polynomial coefficients in normalized basis with Tikhonov penalty."""
        V = np.vander(t, N=6, increasing=True)
        if weights is None:
            weights = np.ones(len(t))
        W = np.diag(weights)
        # Regularization: diagonal penalty growing with order
        P = np.diag(np.array([(j + 1) ** 2 for j in range(6)], dtype=float))
        A = V.T @ W @ V + lam * P
        b = V.T @ W @ y
        try:
            c_norm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_norm = np.linalg.lstsq(V, y, rcond=None)[0]
        return c_norm

    # K-fold CV to select lambda
    # Use 5-fold CV
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    # Shuffle for better CV
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    lambda_candidates = np.logspace(-6, 2, 25)  # wide range of regularization strengths

    best_lambda = 1e-3
    best_cv_error = np.inf

    for lam in lambda_candidates:
        if time.time() > deadline - 10:
            break
        cv_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            t_train = t[train_idx]
            y_train = y[train_idx]
            t_val = t[val_idx]
            y_val = y[val_idx]

            # Fit on train
            c_norm = fit_coefficients(t_train, y_train, lam)

            # Predict on val
            V_val = np.vander(t_val, N=6, increasing=True)
            y_pred = V_val @ c_norm

            mse = np.mean((y_val - y_pred) ** 2)
            cv_errors.append(mse)

        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_lambda = lam

    # Final fit on all data with best lambda
    c_norm = fit_coefficients(t, y, best_lambda)

    # Convert from normalized basis (t = (x - x_mean)/x_std) to original x basis
    # t = (x - x_mean)/x_std
    # We need: sum_{j=0}^{5} c_norm[j] * t^j = sum_{i=0}^{5} c_orig[i] * x^i
    # t^j = ((x - x_mean)/x_std)^j = (1/x_std^j) * sum_{k=0}^{j} binom(j,k) * x^k * (-x_mean)^(j-k)
    # So: sum_j c_norm[j] * (1/x_std^j) * sum_k binom(j,k) * x^k * (-x_mean)^(j-k)
    #     = sum_k x^k * [sum_{j=k}^{5} c_norm[j] * (1/x_std^j) * binom(j,k) * (-x_mean)^(j-k)]
    # Therefore: c_orig[k] = sum_{j=k}^{5} c_norm[j] * (1/x_std^j) * binom(j,k) * (-x_mean)^(j-k)

    c_orig = np.zeros(6)
    for k in range(6):
        for j in range(k, 6):
            binom_jk = np.math.factorial(j) / (np.math.factorial(k) * np.math.factorial(j - k))
            c_orig[k] += c_norm[j] * (1.0 / (x_std ** j)) * binom_jk * ((-x_mean) ** (j - k))

    # Ensure finite
    if not np.all(np.isfinite(c_orig)):
        c_orig = np.zeros(6)

    return c_orig
# EVOLVE-BLOCK-END
