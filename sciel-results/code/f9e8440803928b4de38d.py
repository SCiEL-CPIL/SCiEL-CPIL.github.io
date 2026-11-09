import numpy as np
from scipy.optimize import minimize
import time

def calc_P(support, weights):
    """Compute P = Prob[X1+X2+X3 < 2*X4] for discrete measure."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # S_ij = s_i + s_j, shape (n,n)
    S = s[:, None] + s[None, :]
    # R_kl = 2*s_l - s_k, shape (n,n)
    R = 2.0 * s[None, :] - s[:, None]
    
    # mask[i,j,k,l] = 1(S_ij < R_kl)
    # Use broadcasting: S[:, :, None, None] < R[None, None, :, :]
    mask = (S[:, :, None, None] < R[None, None, :, :])
    
    # W_prod[i,j,k,l] = w_i * w_j * w_k * w_l
    W_prod = w[:, None, None, None] * w[None, :, None, None] * w[None, None, :, None] * w[None, None, None, :]
    
    return float(np.sum(W_prod * mask))

def neg_P(params, n):
    """Negative P for minimization. params = [s0,...,s_{n-1}, w0,...,w_{n-1}]"""
    s = params[:n]
    w = params[n:]
    
    # Ensure s >= 0 and w >= 0 with small penalty
    if np.any(s < -1e-6) or np.any(w < -1e-6):
        return 1e6
    s = np.clip(s, 0, None)
    w = np.clip(w, 0, None)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e6
    w = w / w_sum
    
    return -calc_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    def update(s, w):
        nonlocal best_P, best_support, best_weights
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return
        w = w / w_sum
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if np.any(np.isnan(s)) or np.any(np.isnan(w)):
            return
        P = calc_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Try n = 2, 3, 4, 5, 6
    for n in range(2, 7):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        n_restarts = 20 if n <= 3 else 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initial support in [0, 10]
            s_init = np.sort(np.random.uniform(0, 10, n))
            # Random initial weights (Dirichlet-like)
            w_init = np.random.dirichlet(np.ones(n))
            
            params_init = np.concatenate([s_init, w_init])
            
            try:
                result = minimize(
                    neg_P, params_init, args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                continue
            
            if result.fun < 1e5:  # valid result
                s_opt = np.clip(result.x[:n], 0, None)
                w_opt = np.clip(result.x[n:], 0, None)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                    update(s_opt, w_opt)
            
            # Also try the initial point
            update(s_init, w_init)
    
    # Fallback: binary distribution
    if best_P < 0.25:
        p_opt = 0.5
        s_bin = np.array([0.0, 1.0])
        w_bin = np.array([1-p_opt, p_opt])
        update(s_bin, w_bin)
    
    # Final validation
    if best_support is None or best_weights is None:
        s = np.array([0.0, 1.0])
        w = np.array([0.5, 0.5])
        return s, w
    
    if np.any(best_weights < -1e-9) or np.any(best_support < -1e-9):
        s = np.array([0.0, 1.0])
        w = np.array([0.5, 0.5])
        return s, w
    if np.sum(best_weights) <= 1e-9:
        s = np.array([0.0, 1.0])
        w = np.array([0.5, 0.5])
        return s, w
    if np.any(np.isnan(best_weights)) or np.any(np.isnan(best_support)):
        s = np.array([0.0, 1.0])
        w = np.array([0.5, 0.5])
        return s, w
    if np.any(np.isinf(best_weights)) or np.any(np.isinf(best_support)):
        s = np.array([0.0, 1.0])
        w = np.array([0.5, 0.5])
        return s, w
    
    return best_support, best_weights
