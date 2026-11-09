"""Symbolic regression using Gaussian Process with Cholesky decomposition for noiseless 1D data."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def _rbf_kernel(x1, x2, length_scale, signal_var):
    """Compute RBF kernel matrix between two 1D arrays."""
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=float).reshape(-1, 1)
    sq_dist = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2.0 * x1 @ x2.T
    sq_dist = np.maximum(sq_dist, 0.0)
    return signal_var * np.exp(-sq_dist / (2.0 * length_scale**2))


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    # Constant check
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0

    # Sort by x for better numerical stability
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]

    # GP hyperparameter optimization using log-likelihood
    # Parameters: log(length_scale), log(signal_var), log(noise_var)
    # Since data is noiseless, we use a tiny noise floor for numerical stability
    
    def _log_marginal_likelihood(params):
        """Negative log marginal likelihood for GP with RBF kernel."""
        if time.time() > deadline - 1:
            return 1e10
        
        log_ls, log_sv, log_nv = params
        length_scale = np.exp(log_ls)
        signal_var = np.exp(log_sv)
        noise_var = np.exp(log_nv)
        
        # Ensure numerical stability
        if length_scale < 1e-8 or signal_var < 1e-12 or noise_var < 1e-12:
            return 1e10
        
        try:
            # Compute kernel matrix
            K = _rbf_kernel(x_sorted, x_sorted, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
            
            # Solve L alpha = y
            alpha = np.linalg.solve(L, y_sorted)
            alpha = np.linalg.solve(L.T, alpha)
            
            # Log marginal likelihood
            log_lik = -0.5 * y_sorted @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(log_lik):
                return 1e10
            
            return -log_lik
        except np.linalg.LinAlgError:
            return 1e10
        except Exception:
            return 1e10

    # Initial guesses for hyperparameters
    # length_scale ~ x_range/4, signal_var ~ y_var, noise_var ~ tiny
    init_ls = x_range / 4.0
    init_sv = max(y_var, 1e-6)
    init_nv = 1e-8 * max(y_var, 1e-6)
    
    x0 = np.log([init_ls, init_sv, init_nv])
    
    # Try multiple initializations to avoid local minima
    best_neg_ll = np.inf
    best_params = x0.copy()
    
    # Several starting points
    ls_candidates = [x_range * f for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
    sv_candidates = [y_var * f for f in [0.1, 1.0, 10.0]]
    nv_candidates = [1e-10, 1e-8, 1e-6]
    
    n_inits = 0
    max_inits = 12
    
    for ls_c in ls_candidates[:4]:
        for sv_c in sv_candidates[:2]:
            if n_inits >= max_inits:
                break
            for nv_c in nv_candidates[:2]:
                if n_inits >= max_inits:
                    break
                if time.time() > deadline - 30:
                    break
                
                x_init = np.log([max(ls_c, 1e-6), max(sv_c, 1e-10), max(nv_c, 1e-15)])
                try:
                    result = minimize(
                        _log_marginal_likelihood,
                        x_init,
                        method='L-BFGS-B',
                        bounds=[(np.log(1e-6), np.log(x_range * 5)),
                                (np.log(1e-10), np.log(y_var * 100)),
                                (np.log(1e-15), np.log(max(y_var, 1e-6) * 0.1))],
                        options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
                    )
                    if result.fun < best_neg_ll:
                        best_neg_ll = result.fun
                        best_params = result.x.copy()
                except Exception:
                    pass
                n_inits += 1
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # Extract best hyperparameters
    best_ls = np.exp(best_params[0])
    best_sv = np.exp(best_params[1])
    best_nv = np.exp(best_params[2])
    
    # Ensure reasonable values
    best_ls = np.clip(best_ls, 1e-4, x_range * 5)
    best_sv = np.clip(best_sv, 1e-6, y_var * 100)
    best_nv = np.clip(best_nv, 1e-12, max(y_var, 1e-6) * 0.1)

    # Compute kernel matrix and Cholesky for final model
    try:
        K = _rbf_kernel(x_sorted, x_sorted, best_ls, best_sv)
        K[np.diag_indices_from(K)] += best_nv
        L = np.linalg.cholesky(K)
        
        # Compute alpha = K^{-1} y for prediction
        alpha = np.linalg.solve(L, y_sorted)
        alpha = np.linalg.solve(L.T, alpha)
        
        gp_success = True
    except (np.linalg.LinAlgError, Exception):
        # Fallback: use simpler approach
        gp_success = False
        alpha = None
        L = None

    if not gp_success:
        # Fallback to polynomial fit
        mean_y = np.mean(y_vals)
        try:
            deg = min(5, n - 1)
            pc = np.polyfit(x_vals, y_vals, deg)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
                try:
                    pred = np.polyval(pc, X_pred)
                    pred = np.where(np.isfinite(pred), pred, mean_y)
                    return pred
                except Exception:
                    return np.full(len(X_pred), mean_y)
            return predict
        except Exception:
            return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        try:
            # Compute K_* = kernel(test, train)
            K_star = _rbf_kernel(X_pred, x_sorted, best_ls, best_sv)
            
            # Prediction mean: K_star @ K^{-1} y = K_star @ alpha
            pred_mean = K_star @ alpha
            
            # Clip to finite values
            pred_mean = np.where(np.isfinite(pred_mean), pred_mean, np.mean(y_sorted))
            
            return pred_mean
        except Exception:
            # Fallback to interpolation or mean
            try:
                # Simple linear interpolation as fallback
                pred = np.interp(X_pred, x_sorted, y_sorted)
                pred = np.where(np.isfinite(pred), pred, np.mean(y_sorted))
                return pred
            except Exception:
                return np.full(n_pred, np.mean(y_sorted))

    return predict
# EVOLVE-BLOCK-END
