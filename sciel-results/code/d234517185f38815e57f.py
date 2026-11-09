# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    This implementation uses numerical optimization (L-BFGS-B) to maximize the
    objective function directly. It uses a softmax parameterization to ensure
    valid probabilities and performs multiple random restarts to find a better
    local optimum.
    """
    n = 10
    
    def compute_entropy(probs):
        # probs is a 1D array of pooled probabilities
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def objective(params):
        # params is a flattened 10x10 array of log-weights
        W = np.exp(params).reshape(n, n)
        total = W.sum()
        if total <= 0:
            return 0.0
        P = W / total
        
        # Compute marginals/projections
        # X: sum over y for each x
        p_x = P.sum(axis=0)
        # X+Y: sum over cells where x+y=k
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                p_xy[x + y] += P[y, x]
        # X+2Y: sum over cells where x+2y=k
        p_x2y = np.zeros(3*n - 1) # max index 9 + 2*9 = 27, size 28
        for y in range(n):
            for x in range(n):
                p_x2y[x + 2*y] += P[y, x]
        # X-Y: sum over cells where x-y=k, range -(n-1) to n-1
        p_xm_y = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                # shift index by n-1
                p_xm_y[x - y + (n-1)] += P[y, x]

        H_x = compute_entropy(p_x)
        H_xy = compute_entropy(p_xy)
        H_x2y = compute_entropy(p_x2y)
        H_xm_y = compute_entropy(p_xm_y)
        
        M = max(H_x, H_xy, H_x2y)
        
        if M < 1e-9:
            return 0.0
            
        ratio = H_xm_y / M
        # We want to maximize ratio, so minimize -log(ratio) for stability
        if ratio <= 0:
            return 1000.0 # Large penalty
        return -np.log(ratio)

    best_P = None
    best_score = -np.inf
    
    # Number of restarts to try
    num_restarts = 5
    random.seed(42)
    
    for _ in range(num_restarts):
        # Initialize parameters randomly
        params0 = np.random.randn(n * n) * 0.5
        # Or start with uniform (zeros) for one of the restarts
        if _ == 0:
            params0 = np.zeros(n * n)
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < 1000:
                params_opt = res.x
                W = np.exp(params_opt).reshape(n, n)
                total = W.sum()
                if total > 0:
                    P_candidate = W / total
                    # Verify validity and score
                    score_val = -res.fun
                    if score_val > best_score:
                        # Re-calculate actual score to be sure
                        # (res.fun is -log(ratio), so ratio = exp(-res.fun))
                        # But let's just check if it's better than current best
                        best_P = P_candidate
                        best_score = score_val
        except Exception:
            continue

    # If optimization failed or didn't find anything better than a baseline, 
    # we might want to ensure we return a valid distribution.
    # The previous perimeter heuristic was a safe fallback.
    if best_P is None:
        # Fallback to perimeter
        P = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            P[i, 0] = 1.0
            P[i, n-1] = 1.0
            P[0, i] = 1.0
            P[n-1, i] = 1.0
        total = P.sum()
        if total > 0:
            best_P = P / total
        else:
            best_P = np.ones((n, n), dtype=np.float64) / (n * n)
            
    # Final safety checks
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P /= s
        
    return best_P

# EVOLVE-BLOCK-END
