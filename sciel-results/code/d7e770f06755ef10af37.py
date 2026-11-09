# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten for sorting
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort sums and get corresponding probabilities
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Precompute thresholds: thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find the number of pair_sums strictly less than it
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Now we need the sum of sorted_probs up to indices.
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    
    # Mask for indices > 0
    mask_pos = indices > 0
    
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    
    # Now P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def _optimize_weights_for_support(s, z0=None, maxiter=2000):
    """Optimize weights for a fixed support using softmax parameterization."""
    n = len(s)
    if z0 is None:
        z0 = np.zeros(n)
    
    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        P = _compute_P_fast(s, w)
        return -P
    
    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                       options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        return w_opt, P_opt
    except Exception:
        return None, 0.0

def _joint_optimize(s0, w0, maxiter=500):
    """Jointly optimize support and weights."""
    n = len(s0)
    
    def objective(params):
        # params: [log_s_1, ..., log_s_{n-1}, z_1, ..., z_n]
        # s[0] fixed at 0, s[i] = s[0] + exp(params[i-1]) for i >= 1
        # w from softmax of params[n:]
        log_s = params[:n-1]
        z = params[n-1:]
        
        # Reconstruct support
        s = np.zeros(n)
        s[0] = 0.0
        s[1:] = np.exp(log_s)
        # Ensure sorted (should be if we start sorted and don't cross)
        s = np.sort(s)
        
        # Softmax weights
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        
        P = _compute_P_fast(s, w)
        return -P
    
    # Initial params
    log_s0 = np.log(np.maximum(s0[1:] - s0[0], 1e-6)) if n > 1 else np.array([])
    z0 = np.log(np.maximum(w0, 1e-6))
    z0 = z0 - z0.mean()  # Center for numerical stability
    
    x0 = np.concatenate([log_s0, z0])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-7})
        
        # Extract result
        log_s_opt = res.x[:n-1]
        z_opt = res.x[n-1:]
        
        s_opt = np.zeros(n)
        s_opt[0] = 0.0
        s_opt[1:] = np.exp(log_s_opt)
        s_opt = np.sort(s_opt)
        
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        
        P_opt = -res.fun
        return s_opt, w_opt, P_opt
    except Exception:
        return s0.copy(), w0.copy(), 0.0

def _diff_evolve_support(n, bounds_s, bounds_z, maxiter=100):
    """Use differential evolution to find good support structure."""
    # Parameters: log_s[1..n-1], z[0..n-1]
    # s[0] = 0 fixed
    lb = np.concatenate([bounds_s[0], bounds_z[0]])
    ub = np.concatenate([bounds_s[1], bounds_z[1]])
    
    def objective(params):
        log_s = params[:n-1]
        z = params[n-1:]
        
        s = np.zeros(n)
        s[0] = 0.0
        s[1:] = np.exp(log_s)
        s = np.sort(s)
        
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        
        P = _compute_P_fast(s, w)
        return -P
    
    try:
        res = differential_evolution(objective, zip(lb, ub), maxiter=maxiter, 
                                     seed=42, tol=1e-6, polish=False)
        
        log_s_opt = res.x[:n-1]
        z_opt = res.x[n-1:]
        
        s_opt = np.zeros(n)
        s_opt[0] = 0.0
        s_opt[1:] = np.exp(log_s_opt)
        s_opt = np.sort(s_opt)
        
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        
        P_opt = -res.fun
        return s_opt, w_opt, P_opt
    except Exception:
        return None, None, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    def _update_best(s, w, P):
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Phase 1: Try structured supports with weight optimization
    supports_to_try = []
    
    # Geometric progressions (key insight: geometric decay concentrates mass at small values)
    for n in [3, 4, 5, 6, 8, 10, 12]:
        for base in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            s = np.array([base**i for i in range(n)], dtype=np.float64)
            # Normalize so max is around 10-100
            s = s / s[-1] * 50.0
            supports_to_try.append(s)
            
    # Arithmetic progressions
    for n in [3, 4, 5, 6, 8, 10, 15]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
            
    # Mixed: 0, small cluster, then large jumps
    for n in [5, 6, 8, 10]:
        for k in range(2, min(n, 5)):
            s = np.zeros(n, dtype=np.float64)
            s[:k] = np.arange(k, dtype=np.float64) * 0.5
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1) ** 1.5
            supports_to_try.append(s)
    
    # Log-spaced
    for n in [5, 6, 8, 10]:
        s = np.geomspace(0.1, 100, n)
        supports_to_try.append(s)
    
    # Optimize weights for each support
    for s in supports_to_try:
        if time.time() > deadline - 90.0:
            break
            
        n = len(s)
        w0 = np.ones(n) / n
        
        # Try multiple initializations
        for init_seed in range(3):
            if init_seed == 0:
                z0 = np.zeros(n)
            elif init_seed == 1:
                # Geometric decay weights
                z0 = np.linspace(0, -2, n)
            else:
                # Concentrated at first point
                z0 = np.zeros(n)
                z0[0] = 2.0
            
            w_opt, P_opt = _optimize_weights_for_support(s, z0, maxiter=1000)
            if w_opt is not None and P_opt > best_P:
                _update_best(s, w_opt, P_opt)
    
    # Phase 2: Joint optimization from best structured solution
    if time.time() < deadline - 60.0:
        s0 = best_support.copy()
        w0 = best_weights.copy()
        n = len(s0)
        
        # Ensure s0[0] = 0
        s0 = s0 - s0.min()
        
        # Try joint optimization with different maxiter
        for maxiter in [200, 500, 1000]:
            if time.time() > deadline - 30.0:
                break
            s_opt, w_opt, P_opt = _joint_optimize(s0, w0, maxiter=maxiter)
            if P_opt > best_P:
                _update_best(s_opt, w_opt, P_opt)
                s0 = s_opt.copy()
                w0 = w_opt.copy()
    
    # Phase 3: Differential evolution for support structure
    if time.time() < deadline - 120.0:
        np.random.seed(42)
        
        # Try different n values with DE
        for n in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 60.0:
                break
            
            # Bounds for log_s: s[i] in [0.01, 1000] -> log_s in [-4.6, 6.9]
            bounds_s = (np.full(n-1, -4.6), np.full(n-1, 6.9))
            # Bounds for z: z in [-5, 5]
            bounds_z = (np.full(n, -5.0), np.full(n, 5.0))
            
            s_opt, w_opt, P_opt = _diff_evolve_support(n, bounds_s, bounds_z, maxiter=50)
            if s_opt is not None and P_opt > best_P:
                _update_best(s_opt, w_opt, P_opt)
                
                # Refine with joint optimization
                if time.time() < deadline - 30.0:
                    s_ref, w_ref, P_ref = _joint_optimize(s_opt, w_opt, maxiter=500)
                    if P_ref > best_P:
                        _update_best(s_ref, w_ref, P_ref)
    
    # Phase 4: Random restarts with weight optimization
    if time.time() < deadline - 30.0:
        np.random.seed(123)
        for _ in range(15):
            if time.time() > deadline - 15.0:
                break
            n = np.random.randint(3, 9)
            # Random log-spaced support
            log_s = np.sort(np.random.uniform(-2, 5, n-1))
            s = np.zeros(n)
            s[0] = 0.0
            s[1:] = np.exp(log_s)
            
            w_opt, P_opt = _optimize_weights_for_support(s, np.zeros(n), maxiter=500)
            if w_opt is not None and P_opt > best_P:
                _update_best(s, w_opt, P_opt)
    
    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
        
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
        
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
