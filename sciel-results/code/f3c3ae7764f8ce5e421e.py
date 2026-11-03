import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 630.0  # 600s limit + margin, but we should be fast
    
    def score_for_alpha(alpha):
        # Compute roots of L_12^(alpha)(x)
        try:
            nodes, weights = roots_laguerre(12, alpha)
        except Exception:
            return np.inf
        
        # Filter positive roots (they should be)
        nodes = nodes[nodes > 0]
        if len(nodes) != 12:
            return np.inf
            
        # Proxy score: Minimize the condition number of the Vandermonde matrix
        # Or maximize the product of weights (quadrature stability)
        # Let's use the condition number of the Vandermonde matrix V where V[i,j] = nodes[i]^j
        # High condition number = ill-conditioned = likely worse bound
        V = np.vander(nodes, N=12, increasing=True)
        cond = np.linalg.cond(V)
        
        # Another proxy: The "spread" of the roots. 
        # Often, tighter clustering at the origin is better for Laguerre weights.
        # Let's combine: Minimize cond, but also penalize if roots are too far out.
        max_root = np.max(nodes)
        
        # The C4 bound is often related to the tail behavior.
        # Let's assume the score is inversely proportional to the "quality".
        # We want to MINIMIZE the score (upper bound).
        # If the score is the upper bound, and the upper bound is smaller for "better" roots...
        # Let's try to minimize the condition number.
        return cond
    
    # Search over alpha
    best_score = np.inf
    best_alpha = 0.0
    
    # Grid search first
    alphas = np.linspace(0, 5, 50)
    for a in alphas:
        if time.time() > deadline - 10:
            break
        s = score_for_alpha(a)
        if s < best_score:
            best_score = s
            best_alpha = a
            
    # Local refinement
    def objective(alpha):
        return score_for_alpha(alpha)
        
    try:
        res = minimize(objective, best_alpha, method='Nelder-Mead', options={'maxiter': 100, 'fatol': 1e-6})
        if res.fun < best_score:
            best_alpha = res.x[0]
    except Exception:
        pass

    # Final check and retrieval
    if time.time() > deadline - 5:
        # Fallback to seed if too slow
        return [4.0 * (i + 1) for i in range(12)]

    nodes, _ = roots_laguerre(12, best_alpha)
    return [float(x) for x in nodes]

# EVOLVE-BLOCK-END
