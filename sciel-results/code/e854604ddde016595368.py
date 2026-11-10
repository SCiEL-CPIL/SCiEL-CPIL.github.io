"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fit_ridge(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Convert from centered/scaled basis to original basis
    coeff_orig = np.zeros(degree + 1)
    for k in range(degree + 1):
        if abs(c_centered[k]) < 1e-15:
            continue
        
        a = 1.0 / sigma
        b_const = -mu / sigma
        
        for i in range(k + 1):
            # binom(k, i)
            binom_k_i = 1
            if i > 0:
                num = 1
                for j in range(k):
                    num *= (j + 1)
                den_i = 1
                for j in range(i):
                    den_i *= (j + 1)
                den_ki = 1
                for j in range(k - i):
                    den_ki *= (j + 1)
                binom_k_i = num / (den_i * den_ki)
            
            term_coeff = binom_k_i * (a ** i) * (b_const ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig

def _polyval_orig(x, coeff):
    """Evaluate polynomial with coefficients in original basis."""
    return np.polynomial.polynomial.polyval(x, coeff)

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)
        
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        if len(x) >= 2:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                coeff[0] = np.mean(y)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    n = len(x)
    
    # Default fallback: linear fit
    try:
        default_coeff = np.zeros(6)
        default_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    except:
        default_coeff = np.zeros(6)
        default_coeff[0] = np.mean(y)
    
    if n <= 50:
        # Small sample: use ridge with moderate alpha
        try:
            coeff = _fit_ridge(x, y, 5, 1e-2)
            if np.all(np.isfinite(coeff)):
                return coeff[:6]
        except:
            pass
        return default_coeff

    # Larger sample: use cross-validation to select degree and alpha
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_folds = 5
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    candidates = []
    for deg in [5, 4, 3, 2]:
        for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            candidates.append((deg, alpha))
    
    best_cv_score = np.inf
    best_params = (5, 1e-4)
    
    for deg, alpha in candidates:
        if time.time() > deadline - 10.0:
            break
            
        cv_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            try:
                c_centered = _fit_ridge(x_train, y_train, deg, alpha)
                y_pred = _polyval_orig(x_val, c_centered)
                mse = np.mean((y_val - y_pred)**2)
                cv_errors.append(mse)
            except Exception:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_params = (deg, alpha)
            
    deg, alpha = best_params
    
    # Final fit on all data with best params
    try:
        best_coeff = _fit_ridge(x, y, deg, alpha)
        if len(best_coeff) < 6:
            padded = np.zeros(6)
            padded[:len(best_coeff)] = best_coeff
            best_coeff = padded
        else:
            best_coeff = best_coeff[:6]
    except Exception:
        best_coeff = default_coeff
    
    # Optional: try a local refinement using scipy.optimize.minimize
    # to minimize the training MSE directly, starting from the ridge solution.
    # This can help if the ridge solution is slightly suboptimal for the exact training set.
    if time.time() < deadline - 20.0:
        try:
            def objective(coeffs):
                y_pred = _polyval_orig(x, coeffs)
                return np.mean((y - y_pred)**2)
            
            # Only refine if we have enough data relative to parameters
            if n > 2 * 6:
                res = minimize(objective, best_coeff, method='L-BFGS-B', options={'maxiter': 500})
                if res.success and np.all(np.isfinite(res.x)):
                    # Check if improvement is significant
                    if res.fun < objective(best_coeff) * 0.99:
                        best_coeff = res.x
        except Exception:
            pass
    
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = default_coeff
        
    return best_coeff[:6]
# EVOLVE-BLOCK-END
