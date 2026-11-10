# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    This implementation uses global optimization (Differential Evolution) to find
    a good candidate, followed by local refinement (L-BFGS-B) to maximize the
    objective function. It uses a softmax parameterization to ensure valid 
    probabilities.
    """
    n = 10
    
    def compute_entropy(probs):
        # probs is a 1D array of pooled probabilities
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def get_marginals(P):
        # P is (n, n) where P[y, x]
        # X: sum over y for each x -> shape (n,)
        p_x = P.sum(axis=0)
        
        # X+Y: x+y ranges from 0 to 2n-2
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                p_xy[x + y] += P[y, x]
                
        # X+2Y: x+2y ranges from 0 to 3n-2 (max 9 + 18 = 27)
        p_x2y = np.zeros(3*n - 1)
        for y in range(n):
            for x in range(n):
                p_x2y[x + 2*y] += P[y, x]
                
        # X-Y: x-y ranges from -(n-1) to n-1. Index shift by n-1.
        p_xm_y = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                p_xm_y[x - y + (n-1)] += P[y, x]
                
        return p_x, p_xy, p_x2y, p_xm_y

    def objective(params):
        # params is a flattened 10x10 array of log-weights
        # Clip to prevent overflow in exp
        params_clipped = np.clip(params, -10, 10)
        W = np.exp(params_clipped).reshape(n, n)
        total = W.sum()
        if total <= 1e-15:
            return 0.0
        P = W / total
        
        p_x, p_xy, p_x2y, p_xm_y = get_marginals(P)

        H_x = compute_entropy(p_x)
        H_xy = compute_entropy(p_xy)
        H_x2y = compute_entropy(p_x2y)
        H_xm_y = compute_entropy(p_xm_y)
        
        M = max(H_x, H_xy, H_x2y)
        
        if M < 1e-9:
            return 0.0
            
        ratio = H_xm_y / M
        if ratio <= 0:
            return 1000.0 # Large penalty
        # We want to maximize ratio, so minimize -log(ratio)
        return -np.log(ratio)

    def get_score(params):
        # Returns the actual ratio, not -log(ratio)
        params_clipped = np.clip(params, -10, 10)
        W = np.exp(params_clipped).reshape(n, n)
        total = W.sum()
        if total <= 1e-15:
            return 0.0
        P = W / total
        
        p_x, p_xy, p_x2y, p_xm_y = get_marginals(P)

        H_x = compute_entropy(p_x)
        H_xy = compute_entropy(p_xy)
        H_x2y = compute_entropy(p_x2y)
        H_xm_y = compute_entropy(p_xm_y)
        
        M = max(H_x, H_xy, H_x2y)
        
        if M < 1e-9:
            return 0.0
        return H_xm_y / M

    best_P = None
    best_score = -np.inf
    best_params = None
    
    # 1. Global Search with Differential Evolution
    # Bounds for log-weights
    bounds = [(-10, 10)] * (n * n)
    
    # Differential Evolution is stochastic, so we seed it for reproducibility if needed, 
    # but we want diversity. We'll run it once with a decent population.
    try:
        # popsize=15, maxiter=50 -> 750 * 100 = 75,000 evaluations
        # This is fast. We can afford more.
        res_de = differential_evolution(objective, bounds, maxiter=50, popsize=20, seed=42, polish=False)
        if res_de.fun < 1000:
            best_params = res_de.x
            best_score = get_score(best_params)
    except Exception as e:
        pass

    # 2. Local Refinement with L-BFGS-B
    # If DE found a good starting point, polish it.
    if best_params is not None:
        try:
            res_lbfgs = minimize(objective, best_params, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            if res_lbfgs.fun < 1000:
                params_opt = res_lbfgs.x
                score_val = get_score(params_opt)
                if score_val > best_score:
                    best_params = params_opt
                    best_score = score_val
        except Exception:
            pass
    
    # 3. Additional Random Restarts for Local Refinement
    # Sometimes DE misses the global peak, so we try a few random starts for L-BFGS-B
    random.seed(42)
    np.random.seed(42)
    
    for i in range(5):
        if i == 0:
            # Start near uniform (log-weights near 0)
            params0 = np.random.randn(n * n) * 0.1
        else:
            params0 = np.random.randn(n * n) * 2.0
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < 1000:
                score_val = get_score(res.x)
                if score_val > best_score:
                    best_params = res.x
                    best_score = score_val
        except Exception:
            continue

    # Construct final P
    if best_params is not None:
        params_clipped = np.clip(best_params, -10, 10)
        W = np.exp(params_clipped).reshape(n, n)
        total = W.sum()
        if total > 1e-15:
            best_P = W / total
        else:
            best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        # Fallback to perimeter if everything failed
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
