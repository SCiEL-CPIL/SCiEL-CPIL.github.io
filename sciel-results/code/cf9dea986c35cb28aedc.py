# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_entropies(P):
    """Compute H_{-1}, H_0, H_1, H_2 for a given P."""
    n = P.shape[0]
    # Pre-compute indices for slopes -1, 0, 1, 2
    # We can use a dictionary or just sum over masks.
    # For efficiency, we can precompute the masks or use bincount.
    
    def entropy_from_proj(P, slope):
        # P is n x n, P[y, x]
        # val = x + slope * y
        # Create a 2D array of values
        y_idx, x_idx = np.indices((n, n))
        vals = x_idx + slope * y_idx
        # Flatten
        vals_flat = vals.flatten()
        p_flat = P.flatten()
        # Use bincount to sum probabilities for each unique value
        # vals range: 
        # slope -1: x - y. min -9, max 9. range 19.
        # slope 0: x. min 0, max 9. range 10.
        # slope 1: x + y. min 0, max 18. range 19.
        # slope 2: x + 2y. min 0, max 27. range 28.
        
        # To be general and safe, use unique
        unique_vals, inverse = np.unique(vals_flat, return_inverse=True)
        probs = np.bincount(inverse, weights=p_flat)
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    D = entropy_from_proj(P, -1)
    H0 = entropy_from_proj(P, 0)
    H1 = entropy_from_proj(P, 1)
    H2 = entropy_from_proj(P, 2)
    M = max(H0, H1, H2)
    return D, M

def _objective(theta):
    """Objective function for minimization. Minimizes -log(D/M) = -log(D) + log(M)."""
    # Softmax to get P
    # theta is flattened 100x1
    theta = theta - np.max(theta) # for numerical stability
    exp_theta = np.exp(theta)
    P = exp_theta / np.sum(exp_theta)
    
    D, M = _compute_entropies(P.reshape(10, 10))
    
    if D < 1e-15 or M < 1e-15:
        return 1e10 # Large penalty
    
    # We want to maximize D/M, so minimize -log(D/M)
    # If D/M is large, -log(D/M) is negative.
    return -np.log(D) + np.log(M)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 620
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = 0.0
    
    # Define multiple initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros((n, n)))
    
    # 2. Diagonal heavy
    P_diag = np.zeros((n, n))
    np.fill_diagonal(P_diag, 1.0)
    inits.append(np.log(P_diag + 1e-15))
    
    # 3. Anti-diagonal heavy
    P_anti = np.zeros((n, n))
    for i in range(n):
        P_anti[i, n-1-i] = 1.0
    inits.append(np.log(P_anti + 1e-15))
    
    # 4. Corner heavy (top-left, bottom-right)
    P_corner = np.zeros((n, n))
    P_corner[0,0] = 0.5
    P_corner[n-1,n-1] = 0.5
    inits.append(np.log(P_corner + 1e-15))
    
    # 5. Random sparse
    np.random.seed(42)
    P_rand = np.random.rand(n, n)
    P_rand = P_rand / P_rand.sum()
    inits.append(np.log(P_rand + 1e-15))
    
    # 6. Random dense
    np.random.seed(123)
    P_rand2 = np.random.rand(n, n)
    P_rand2 = P_rand2 / P_rand2.sum()
    inits.append(np.log(P_rand2 + 1e-15))

    for init in inits:
        if time.time() > deadline:
            break
            
        theta_init = init.flatten()
        
        try:
            # Use L-BFGS-B as it is efficient for large-ish unconstrained problems
            # SLSQP might be slower but more robust for non-smoothness? 
            # The objective is smooth except where D or M is 0 or where max switches.
            # L-BFGS-B should handle it well.
            res = minimize(_objective, theta_init, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            theta_opt = res.x
            # Reconstruct P
            theta_opt = theta_opt - np.max(theta_opt)
            exp_theta = np.exp(theta_opt)
            P_opt = exp_theta / np.sum(exp_theta)
            P_opt = P_opt.reshape(n, n)
            
            # Evaluate score
            D, M = _compute_entropies(P_opt)
            if M > 1e-9:
                score = D / M
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_P = P_opt.copy()
                
        except Exception:
            continue
            
    # Final check and cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
