import numpy as np
from scipy.optimize import minimize
import time

def compute_P(s, w):
    """
    Computes P[X1 + X2 + X3 < 2*X4] efficiently.
    s: support points (n,)
    w: weights (n,)
    """
    n = len(s)
    if n == 0:
        return 0.0
        
    # Pairwise sums S_ij = s_i + s_j
    S_ij = s[:, np.newaxis] + s[np.newaxis, :]
    # Pairwise weight products W_ij = w_i * w_j
    W_ij = w[:, np.newaxis] * w[np.newaxis, :]
    
    # Flatten and sort
    S_flat = S_ij.flatten()
    W_flat = W_ij.flatten()
    
    # Sort by S_ij
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative sum of weights
    W_cumsum = np.cumsum(W_sorted)
    
    P = 0.0
    for l in range(n):
        # Thresholds T_k = 2*s_l - s_k for all k
        T_k = 2 * s[l] - s
        # Find number of pairs (i,j) such that S_ij < T_k
        # np.searchsorted returns index of first element >= T_k
        # So index is the count of elements < T_k
        counts = np.searchsorted(S_sorted, T_k, side='left')
        # The sum of W_ij for S_ij < T_k is W_cumsum[counts - 1]
        # Need to handle counts == 0
        Q_l = np.zeros(n)
        mask = counts > 0
        Q_l[mask] = W_cumsum[counts[mask] - 1]
        
        # P_l = sum_k w_k * Q_l[k]
        P_l = np.dot(w, Q_l)
        P += w[l] * P_l
        
    return P

def objective(params, n):
    """
    Objective function for minimization (maximizing P).
    params: concatenated vector of u (for s) and v (for w)
    n: number of support points
    """
    u = params[:n]
    v = params[n:]
    
    # s = u^2 ensures s >= 0
    s = u ** 2
    # w = softmax(v) ensures w > 0 and sum(w) = 1
    v_max = np.max(v)
    exp_v = np.exp(v - v_max)
    w = exp_v / np.sum(exp_v)
    
    # Check for numerical issues
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        return 1e10
        
    P = compute_P(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590  # Keep 10s margin
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    # Try different n values
    # Small n is faster to evaluate, allowing more optimization steps
    # The optimal distribution might be sparse or have few points
    n_values = [8, 12, 16]
    
    rng = np.random.default_rng(42)
    
    for n in n_values:
        if time.time() > deadline - 60:
            break
            
        # Perform multiple random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
                
            # Initialize parameters
            # u for s: small random values, squared later
            u0 = rng.uniform(0, 1, n)
            # v for w: random values for softmax
            v0 = rng.uniform(0, 1, n)
            
            params0 = np.concatenate([u0, v0])
            
            # Nelder-Mead optimization
            # maxiter is limited to prevent hanging
            res = minimize(
                objective, 
                params0, 
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if res.success or res.fun < -best_P:
                # Extract s and w
                u_opt = res.x[:n]
                v_opt = res.x[n:]
                
                s_opt = u_opt ** 2
                v_max = np.max(v_opt)
                exp_v = np.exp(v_opt - v_max)
                w_opt = exp_v / np.sum(exp_v)
                
                # Calculate P
                P_opt = -res.fun
                
                if P_opt > best_P:
                    best_P = P_opt
                    best_s = s_opt.copy()
                    best_w = w_opt.copy()
                    
                    # If we found a very good P, we might stop early?
                    # P is a probability, max 1.0.
                    # If P > 0.99, it's likely close to optimal.
                    if best_P > 0.99:
                        return best_s, best_w

    # Fallback if no valid result found or time ran out
    if best_s is None:
        best_s = np.array([0.0, 1.0])
        best_w = np.array([0.5, 0.5])
        
    return best_s, best_w
