"""Tikhonov-regularized degree-five polynomial closure with CV-selected lambda."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _fit_regularized(V, y, lam):
    """Solve (V^T V + lam*I) c = V^T y for polynomial coefficients."""
    A = V.T @ V + lam * np.eye(V.shape[1])
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _predict(V, c):
    return V @ c


def _transform_coeffs(coeff_norm, mu, sig):
    """Transform coefficients from normalized x to original x.
    
    If x_norm = (x - mu)/sig, then:
    sum_{i=0}^{5} c_i * x_norm^i = sum_{j=0}^{5} d_j * x^j
    
    where d_j = sum_{i=j}^{5} c_i * C(i,j) * (-mu)^(i-j) / sig^i
    """
    coeff = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_ij = comb(i, j)
            s += coeff_norm[i] * binom_ij * ((-mu) ** (i - j)) / (sig ** i)
        coeff[j] = s
    return coeff


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(training_x)
    if n < 10:
        # Fallback: just fit a linear polynomial
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        return coeff

    # Normalize x to improve conditioning (standardize)
    x_mean = np.mean(training_x)
    x_std = np.std(training_x)
    if x_std < 1e-12:
        x_norm = np.zeros_like(training_x)
    else:
        x_norm = (training_x - x_mean) / x_std

    powers = np.arange(6)
    V_norm = np.power(x_norm[:, None], powers[None, :])

    # K-fold cross-validation to select lambda
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_sizes = np.array_split(indices, n_folds)

    # Candidate lambdas: log-spaced from small to large
    # Use trace(V^T V)/n as a natural scale for the regularization
    trace_vtv = np.trace(V_norm.T @ V_norm)
    lam_scale = trace_vtv / n if trace_vtv > 0 else 1.0
    
    # Also consider the magnitude of y for scaling
    y_std = np.std(training_subgrid_tendency)
    if y_std < 1e-12:
        y_std = 1.0
    
    # Lambda candidates should be comparable to the scale of the Gram matrix
    # We want lam to range from near-zero to something that dominates the Gram matrix
    lambda_candidates = np.concatenate([
        [0.0],
        np.logspace(-8, 2, 40) * lam_scale
    ])

    best_lam = 0.0
    best_cv_score = np.inf

    for lam in lambda_candidates:
        if _time_remaining() < 3.0:
            break
        mse_sum = 0.0
        for fold in range(n_folds):
            val_idx = fold_sizes[fold]
            train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])

            V_train = V_norm[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V_norm[val_idx]
            y_val = training_subgrid_tendency[val_idx]

            c = _fit_regularized(V_train, y_train, lam)
            pred = _predict(V_val, c)
            mse_sum += np.mean((pred - y_val) ** 2)

        cv_score = mse_sum / n_folds
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lam = lam

    # Final fit on all training data with best lambda
    coeff_norm = _fit_regularized(V_norm, training_subgrid_tendency, best_lam)

    # High-precision polish: try nearby lambdas for fine-tuning
    if _time_remaining() > 5.0:
        # Create a finer grid around the best lambda
        if best_lam > 0:
            polish_lams = np.logspace(np.log10(best_lam) - 0.5, np.log10(best_lam) + 0.5, 15)
        else:
            polish_lams = np.linspace(0, lam_scale * 0.1, 15)
        
        # Also include 0 and the original best
        polish_lams = np.unique(np.concatenate([[0.0], [best_lam], polish_lams]))
        
        best_polish_score = np.inf
        best_polish_coeff = coeff_norm
        
        for plam in polish_lams:
            if _time_remaining() < 2.0:
                break
            c = _fit_regularized(V_norm, training_subgrid_tendency, plam)
            # Use a combination of training MSE and coefficient norm as proxy
            pred = _predict(V_norm, c)
            train_mse = np.mean((pred - training_subgrid_tendency) ** 2)
            # The regularization penalty is already implicitly in the solution,
            # but we can use a smooth score that balances fit and smoothness
            coeff_norm_sq = np.sum(c ** 2)
            score = train_mse + plam * coeff_norm_sq / (n * y_std ** 2 + 1e-10)
            
            if score < best_polish_score:
                best_polish_score = score
                best_polish_coeff = c
        
        coeff_norm = best_polish_coeff

    # Transform coefficients from normalized x to original x
    if x_std < 1e-12:
        coeff = np.zeros(6)
        coeff[0] = np.mean(training_subgrid_tendency)
        return coeff

    mu = x_mean
    sig = x_std
    coeff = _transform_coeffs(coeff_norm, mu, sig)

    # Sanity check: ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        except:
            pass
        if not np.all(np.isfinite(coeff)):
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
