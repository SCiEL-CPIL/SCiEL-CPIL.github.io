"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def _fit_poly_scaled(x, y, degree, lam):
    """
    Fit polynomial in scaled basis with Tikhonov regularization.
    Returns coefficients in scaled basis (ascending powers of x_scaled).
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    # Heavier regularization for higher-order terms
    reg_weights = np.array([1.0, 1.0, 50.0, 500.0, 5000.0, 50000.0])[:degree + 1]
    R = np.diag(lam * reg_weights)
    
    try:
        A = V.T @ V + R
        b = V.T @ y
        c_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            c_scaled = np.zeros(degree + 1)
    
    return c_scaled, x_mean, x_std


def _scaled_to_monomial(c_scaled, x_mean, x_std, degree):
    """Convert coefficients from scaled basis to monomial basis in original x."""
    coeff_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        if abs(c_scaled[j]) < 1e-15:
            continue
        scale_factor = c_scaled[j] / (x_std ** j)
        for k in range(j + 1):
            coeff_original[k] += scale_factor * comb(j, k) * ((-x_mean) ** (j - k))
    return coeff_original


def _l1_threshold_fit(x, y, degree, threshold, max_iter=50):
    """
    Iterative L1 thresholding (soft thresholding on scaled-basis coefficients).
    Returns monomial coefficients.
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    reg_weights = np.array([1.0, 1.0, 50.0, 500.0, 5000.0, 50000.0])[:degree + 1]
    
    # Initialize with unregularized least squares
    try:
        c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        c_scaled = np.zeros(degree + 1)
    
    for _ in range(max_iter):
        # Soft threshold
        for i in range(degree + 1):
            t = threshold * reg_weights[i]
            if c_scaled[i] > t:
                c_scaled[i] -= t
            elif c_scaled[i] < -t:
                c_scaled[i] += t
            else:
                c_scaled[i] = 0.0
        
        # Re-fit with zeroed coefficients kept
        active = np.abs(c_scaled) > threshold * 0.1
        if np.sum(active) <= degree + 1:
            if np.sum(active) >= 1:
                V_active = V[:, active]
                try:
                    c_active = np.linalg.lstsq(V_active, y, rcond=None)[0]
                except Exception:
                    c_active = np.zeros(np.sum(active))
                c_new = np.zeros(degree + 1)
                c_new[active] = c_active
                # Re-apply threshold
                for i in range(degree + 1):
                    t = threshold * reg_weights[i]
                    if c_new[i] > t:
                        c_new[i] -= t
                    elif c_new[i] < -t:
                        c_new[i] += t
                    else:
                        c_new[i] = 0.0
                if np.allclose(c_new, c_scaled, atol=threshold * 0.01):
                    c_scaled = c_new
                    break
                c_scaled = c_new
            else:
                c_scaled = np.zeros(degree + 1)
                break
    
    return _scaled_to_monomial(c_scaled, x_mean, x_std, degree)


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit
    
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter non-finite
    mask = np.isfinite(training_x) & np.isfinite(training_subgrid_tendency)
    x = training_x[mask]
    y = training_subgrid_tendency[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if n < 10:
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, min(1, n-1))
            if not np.all(np.isfinite(coeff)):
                coeff[0] = np.mean(y)
            return coeff
        except Exception:
            return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    # Internal cross-validation
    split = int(n * 0.8)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    
    best_coeff_monomial = None
    best_val_mse = np.inf
    
    # Strategy 1: Tikhonov regularization with varying lambda
    lambdas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    for lam in lambdas:
        if time.time() > deadline - 15.0:
            break
        
        try:
            c_scaled, x_mean, x_std = _fit_poly_scaled(x_train, y_train, degree=5, lam=lam)
            
            if not np.all(np.isfinite(c_scaled)):
                continue
            
            x_val_scaled = (x_val - x_mean) / x_std
            V_val = np.vander(x_val_scaled, N=6, increasing=True)
            pred_val = V_val @ c_scaled
            val_mse = np.mean((pred_val - y_val) ** 2)
            
            if not np.isfinite(val_mse):
                continue
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff_monomial = _scaled_to_monomial(c_scaled, x_mean, x_std, degree=5)
        except Exception:
            continue
    
    # Strategy 2: L1 thresholding for sparse selection
    if time.time() < deadline - 10.0:
        thresholds = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        for thresh in thresholds:
            if time.time() > deadline - 10.0:
                break
            
            try:
                coeff_sparse = _l1_threshold_fit(x_train, y_train, degree=5, threshold=thresh)
                
                if not np.all(np.isfinite(coeff_sparse)):
                    continue
                
                # Evaluate on validation
                pred_val = (coeff_sparse[0] + coeff_sparse[1]*x_val + 
                           coeff_sparse[2]*x_val**2 + coeff_sparse[3]*x_val**3 +
                           coeff_sparse[4]*x_val**4 + coeff_sparse[5]*x_val**5)
                val_mse = np.mean((pred_val - y_val) ** 2)
                
                if not np.isfinite(val_mse):
                    continue
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_coeff_monomial = coeff_sparse
            except Exception:
                continue
    
    # Fallback to linear fit
    if best_coeff_monomial is None or not np.all(np.isfinite(best_coeff_monomial)):
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            if not np.all(np.isfinite(coeff)):
                coeff[0] = np.mean(y)
            return coeff
        except Exception:
            return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    if not np.all(np.isfinite(best_coeff_monomial)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = np.mean(y)
        return coeff
    
    return best_coeff_monomial
# EVOLVE-BLOCK-END
