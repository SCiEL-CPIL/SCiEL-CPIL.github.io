from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time

def _area_ref(x, n=64, M=20000):
    """Correct area via full sort at each height. O(M n log n)."""
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    Ls = np.sort(L, axis=0)
    d = Ls[1:, :] - Ls[:-1, :]
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def _soft_min_area(x, n=64, M=20000, beta=50.0):
    """
    Smooth surrogate for the area using soft-min.
    min(a,b) ≈ -log(exp(-beta*a) + exp(-beta*b)) / beta
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Soft-min of d and w: -log(exp(-beta*d) + exp(-beta*w))/beta
    # Use log-sum-exp trick for stability
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    # min(a,b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # = -1/beta * (max + log(exp(-(beta*a - max)) + exp(-(beta*b - max))))
    # where max = max(beta*a, beta*b)
    mx = np.maximum(beta_d, beta_w)
    sm = -mx / beta - np.log(np.exp(-beta_d - mx) + np.exp(-beta_w - mx)) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _soft_min_area_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute soft-min area and its gradient.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted, order = np.sort(L, axis=0), np.argsort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    mx = np.maximum(beta_d, beta_w)
    
    # Soft-min value
    exp_neg_d = np.exp(-beta_d - mx)
    exp_neg_w = np.exp(-beta_w - mx)
    sm = -mx / beta - np.log(exp_neg_d + exp_neg_w) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = float(np.mean(union_lengths))
    
    # Gradient: d/dx_i of sm
    # sm = -1/beta * log(exp(-beta*d) + exp(-beta*w))
    # d/dx_i sm = -1/beta * [(-beta * d/dx_i d) * exp(-beta*d) + 0] / (exp(-beta*d) + exp(-beta*w))
    #           = d/dx_i d * exp(-beta*d) / (exp(-beta*d) + exp(-beta*w))
    #           = d/dx_i d * sigmoid(beta*(w-d))
    # where d/dx_i d = a_{order[j+1,m]} - a_{order[j,m]} for the gap j at height m
    # (since d_j = L_sorted[j+1] - L_sorted[j], and L_i = x_i + a_i*y)
    
    # sigmoid(beta*(w - d))
    arg = beta * (w[np.newaxis, :] - d)
    sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -500, 500)))
    
    # sig has shape (n-1, M)
    # For gap j at height m: d/dx_i d_j = a_{order[j+1,m]} - a_{order[j,m]}
    # grad[i] = sum_m sig[j,m] * (a_{order[j+1,m]} - a_{order[j,m]}) * I(i == order[j+1,m])
    #          + sum_m sig[j,m] * (a_{order[j+1,m]} - a_{order[j,m]}) * I(i == order[j,m]) * (-1)
    
    da = a[order[1:, :]] - a[order[:-1, :]]  # shape (n-1, M)
    contrib = sig * da  # shape (n-1, M)
    
    grad = np.zeros(n, dtype=np.float64)
    # For each gap j and height m, add contrib[j,m] to grad[order[j+1,m]] and subtract from grad[order[j,m]]
    j_idx = np.arange(n - 1)[:, np.newaxis]
    m_idx = np.arange(M)[np.newaxis, :]
    upper = order[j_idx + 1, m_idx]  # shape (n-1, M)
    lower = order[j_idx, m_idx]  # shape (n-1, M)
    
    np.add.at(grad, upper.ravel(), contrib.ravel())
    np.add.at(grad, lower.ravel(), -contrib.ravel())
    
    grad /= M
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start = time.time()
    deadline = start + 600.0

    i_idx = np.arange(n)
    a = (i_idx + 1) / n

    def objective(x):
        return _area_ref(x, n, M)

    # Seed candidates
    seeds = [
        -a * 0.5,
        np.zeros(n, dtype=np.float64),
        -a * 0.4,
        -a * 0.6,
        -a * 0.5 + np.random.default_rng(1).normal(0, 0.1, n),
    ]
    
    best_x = None
    best_val = np.inf

    for s in seeds:
        val = objective(s)
        if val < best_val:
            best_val = val
            best_x = s.copy()
            
        if time.time() > deadline - 30:
            break

    # Optimize from the best seed
    if best_x is not None and time.time() < deadline - 30:
        # Use a smaller M for optimization to speed up?
        # _area_ref uses M=20000.
        # Let's create a wrapper that uses M=2000 for speed during optimization
        M_opt = 2000
        def objective_opt(x):
            return _area_ref(x, n, M_opt)
            
        res = minimize(objective_opt, best_x, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success and res.fun < best_val:
            # Refine with full M
            x_refined = res.x
            # Do a few more steps with full M?
            # Or just evaluate.
            val_full = objective(x_refined)
            if val_full < best_val:
                best_val = val_full
                best_x = x_refined.copy()

    # Final validation
    if best_x is None or not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    
    # One last check with full M
    final_val = _area_ref(best_x, n, M)
    if not np.isfinite(final_val) or final_val <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
