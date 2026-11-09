import numpy as np
import time
from scipy.optimize import minimize

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4]
    """
    n = len(support)
    # Normalize weights
    w = weights / np.sum(weights)
    
    # Precompute pairwise sums for X1+X2 (size n x n)
    # S_12 = s_i + s_j
    S_12 = support[:, None] + support[None, :]
    
    # For each X4 (index k), we need P[X1+X2 < 2*s_k - X3]
    # This is sum_{i,j,l} w_i w_j w_l I(s_i + s_j + s_l < 2 s_k)
    # Let's loop over k (X4)
    P_total = 0.0
    for k in range(n):
        target = 2 * support[k]
        # We need sum_{l} w_l * P[X1+X2 < target - s_l]
        # P[X1+X2 < Y] = sum_{i,j} w_i w_j I(s_i+s_j < Y)
        
        # Vectorized over l?
        # For each l, condition is S_12 < target - support[l]
        # Let's compute counts for each l
        # S_12 is n x n.
        # target - support[l] is scalar for each l.
        
        # Efficient way:
        # Sort S_12? No, weights are involved.
        
        # Loop over l might be slow if n is large, but n=32 is small.
        # n^3 = 32^3 = 32768 operations per k.
        # Total n^4 = 1M operations. Very fast.
        
        prob_k = 0.0
        for l in range(n):
            thresh = target - support[l]
            # Mask for S_12 < thresh
            mask = S_12 < thresh
            # Sum of w_i w_j for i,j where mask is true
            # sum_{i,j} w_i w_j mask[i,j]
            # = sum_i w_i (sum_j w_j mask[i,j])
            # Or just np.sum(w[:, None] * w[None, :] * mask)
            prob_l = np.sum(w[:, None] * w[None, :] * mask)
            prob_k += w[l] * prob_l
        P_total += w[k] * prob_k
        
    return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 580.0 # Safety margin
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # Candidate supports
    supports = []
    
    # 1. Uniform small
    supports.append((np.arange(10, dtype=np.float64), np.ones(10)/10))
    # 2. Uniform medium (Seed)
    supports.append((np.arange(32, dtype=np.float64), np.ones(32)/32))
    # 3. Uniform large
    supports.append((np.arange(64, dtype=np.float64), np.ones(64)/64))
    # 4. Geometric
    supports.append((np.power(2, np.arange(10, dtype=np.float64)), np.ones(10)/10))
    # 5. Logarithmic
    supports.append((np.log(np.arange(1, 21, dtype=np.float64)), np.ones(20)/20))
    # 6. Sparse high/low (Two-point-like with intermediate)
    supports.append((np.array([0, 1, 10, 100, 1000], dtype=np.float64), np.ones(5)/5))
    # 7. Heavy tail
    supports.append((np.array([0, 1, 5, 20, 80, 320], dtype=np.float64), np.ones(6)/6))
    
    for s_init, w_init in supports:
        n = len(s_init)
        
        # Objective function for minimize (maximize P)
        def objective(w_flat):
            w = w_flat / np.sum(w_flat) # Ensure non-negative and normalized?
            # Handle negative weights by clipping?
            # Better to use simplex constraint.
            # For simplicity, clip and normalize.
            w = np.clip(w, 0, 1)
            if np.sum(w) == 0: return 0.0
            w = w / np.sum(w)
            return -calculate_P(s_init, w)
        
        # Constraints: sum(w) = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * n
        
        # Try a few random restarts or just local search from init
        w_opt = w_init
        best_local_P = calculate_P(s_init, w_init)
        
        # Gradient Ascent (simple)
        # Or use scipy
        try:
            res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                w_opt = res.x
                P_opt = calculate_P(s_init, w_opt)
                if P_opt > best_local_P:
                    best_local_P = P_opt
        except Exception:
            pass
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_support = s_init
            best_weights = w_opt
            
        if time.time() > deadline:
            break
            
    if best_support is None:
        # Fallback to seed
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
        
    return best_support, best_weights
