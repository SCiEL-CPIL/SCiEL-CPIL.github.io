"""Rewritten program using MODEL-AND-OPTIMIZE: fit full degree-5 polynomial with regularization in Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses Chebyshev basis for numerical stability.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (and 150s kill)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    
    # Map x to t in [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min -= 1.0
        x_max += 1.0
    
    x_center = (x_max + x_min) / 2.0
    x_scale = (x_max - x_min) / 2.0
    t = (x - x_center) / x_scale
    
    # Build Chebyshev design matrix
    def build_chebyshev_matrix(t, degree=5):
        """Build Chebyshev matrix T_j(t)."""
        C = np.zeros((len(t), degree + 1))
        if degree >= 0:
            C[:, 0] = 1.0
        if degree >= 1:
            C[:, 1] = t
        for j in range(2, degree + 1):
            C[:, j] = 2.0 * t * C[:, j-1] - C[:, j-2]
        return C
    
    C = build_chebyshev_matrix(t, 5)
    
    # K-fold cross-validation for lambda selection
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    # Use 5-fold CV if enough data, otherwise fewer folds
    if n >= 30:
        n_folds = 5
    elif n >= 12:
        n_folds = 3
    else:
        n_folds = 2
    
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        folds.append(indices[start:end])
    
    # Regularization: uniform weight is sufficient for Chebyshev basis
    reg_weights = np.ones(6)
    reg_matrix = np.diag(reg_weights)
    
    # Log-spaced regularization values
    lambdas = [0.0] + list(np.logspace(-12, 4, 25))
    
    best_mse = np.inf
    best_coeffs_cheb = None
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        try:
            mse_sum = 0.0
            count = 0
            valid_fold = True
            for f in range(n_folds):
                val_idx = folds[f]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
                
                C_train = C[train_idx]
                y_train = y[train_idx]
                C_val = C[val_idx]
                y_val = y[val_idx]
                
                if lam == 0.0:
                    coeffs_lam, _, _, _ = np.linalg.lstsq(C_train, y_train, rcond=None)
                else:
                    CtC = C_train.T @ C_train
                    Cty = C_train.T @ y_train
                    A = CtC + lam * reg_matrix
                    try:
                        coeffs_lam = np.linalg.solve(A, Cty)
                    except np.linalg.LinAlgError:
                        coeffs_lam, _, _, _ = np.linalg.lstsq(A, Cty, rcond=None)
                
                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break
                
                y_val_pred = C_val @ coeffs_lam
                mse_sum += np.mean((y_val_pred - y_val) ** 2)
                count += 1
            
            if not valid_fold or count != n_folds:
                continue
            
            avg_mse = mse_sum / n_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_coeffs_cheb = coeffs_lam.copy()
        except:
            continue
    
    if best_coeffs_cheb is None:
        # Fallback to OLS on all data
        try:
            coeffs_cheb, _, _, _ = np.linalg.lstsq(C, y, rcond=None)
            if np.all(np.isfinite(coeffs_cheb)):
                best_coeffs_cheb = coeffs_cheb
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert Chebyshev coefficients to monomial coefficients in x
    # using stable recurrence to avoid catastrophic cancellation
    
    # M_j is the coefficient vector for T_j(x) in monomial basis
    # T_0(x) = 1
    # T_1(x) = x
    # T_{j+1}(x) = 2x T_j(x) - T_{j-1}(x)
    
    c_cheb = best_coeffs_cheb
    coeff_orig = np.zeros(6)
    
    # Build monomial coefficients for each Chebyshev polynomial T_j(t)
    # T_0(t) = 1 -> [1, 0, 0, 0, 0, 0]
    # T_1(t) = t -> [0, 1, 0, 0, 0, 0]
    # T_2(t) = 2t^2 - 1 -> [-1, 0, 2, 0, 0, 0]
    # T_3(t) = 4t^3 - 2t -> [0, -2, 0, 4, 0, 0]
    # T_4(t) = 8t^4 - 8t^2 + 1 -> [1, 0, -8, 0, 8, 0]
    # T_5(t) = 16t^5 - 20t^3 + 5t -> [0, 5, 0, -20, 0, 16]
    
    # Use recurrence: M_{j+1} = 2x*M_j - M_{j-1}
    # where x*M means shift coefficients right by 1
    
    M_prev = np.array([1.0, 0, 0, 0, 0, 0])  # T_0
    M_curr = np.array([0, 1.0, 0, 0, 0, 0])  # T_1
    
    coeff_orig += c_cheb[0] * M_prev
    if 1 < 6:
        coeff_orig += c_cheb[1] * M_curr
    
    for j in range(2, 6):
        M_next = np.zeros(6)
        # 2x * M_curr: shift right by 1 and multiply by 2
        for k in range(5):
            M_next[k+1] += 2.0 * M_curr[k]
        # subtract M_prev
        M_next -= M_prev
        
        coeff_orig += c_cheb[j] * M_next
        
        M_prev, M_curr = M_curr, M_next
    
    # Now convert monomial coeffs in t to monomial coeffs in x
    # t = (x - x_center) / x_scale = (1/x_scale) * x - (x_center/x_scale)
    # Let u = 1/x_scale, v = -x_center/x_scale
    # t = u*x + v
    # P(x) = sum_{i=0}^5 d_t[i] * (u*x + v)^i
    # Expand (u*x + v)^i = sum_{k=0}^i comb(i, k) * (u*x)^k * v^(i-k)
    #            = sum_{k=0}^i comb(i, k) * u^k * v^(i-k) * x^k
    
    u = 1.0 / x_scale
    v = -x_center / x_scale
    
    d_t = coeff_orig  # monomial coeffs in t
    coeff_orig = np.zeros(6)
    for i in range(6):
        if d_t[i] == 0.0:
            continue
        for k in range(i + 1):
            coeff_orig[k] += d_t[i] * comb(i, k) * (u ** k) * (v ** (i - k))
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
