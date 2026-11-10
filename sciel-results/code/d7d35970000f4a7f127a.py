import numpy as np
from scipy.optimize import differential_evolution

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    n = 3
    
    def calc_P(params):
        # params: [s0, s1, s2, w0, w1, w2]
        s = params[:n]
        w = params[n:]
        
        # Check validity
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 1.0 # Penalty for invalid (minimizing, so high value)
            
        # Normalize weights
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return 1.0
            
        w_norm = w / w_sum
        
        # Compute P
        # P = sum_k sum_l w_k w_l * (sum_i sum_j w_i w_j * I(s_i + s_j < 2*s_l - s_k))
        # Note: indices in prompt:
        # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        # Here X4 is l, X3 is k.
        
        P_val = 0.0
        for k in range(n):
            for l in range(n):
                rhs = 2.0 * s[l] - s[k]
                prob_pair = 0.0
                for i in range(n):
                    for j in range(n):
                        if s[i] + s[j] < rhs:
                            prob_pair += w_norm[i] * w_norm[j]
                P_val += w_norm[k] * w_norm[l] * prob_pair
                
        return -P_val # Maximize P, so minimize -P

    # Bounds for differential evolution
    # s_i >= 0, w_i >= 0
    # We don't strictly need to bound s_i to a max, but it helps convergence.
    # The optimal s might be large. Let's try [0, 1000].
    # Weights sum to 1, so individual max is 1.
    bounds = [(0, 1000), (0, 1000), (0, 1000), 
              (0, 1), (0, 1), (0, 1)]
    
    # Differential Evolution settings
    # maxiter and popsize control runtime.
    # n=3 support, 6 params.
    # 600s limit is generous.
    res = differential_evolution(
        calc_P, 
        bounds, 
        maxiter=200, 
        popsize=15, 
        seed=42,
        tol=1e-6,
        polish=True # Use L-BFGS-B at the end for local refinement
    )
    
    if not res.success and res.fun > 0.5: # If it failed badly or returned penalty
         # Fallback to a known good distribution
         # e.g. the one from current program
         support = np.array([0.0, 1.0, 1000.0], dtype=np.float64)
         weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
         return support, weights

    params = res.x
    s = params[:n]
    w = params[n:]
    
    # Final cleanup
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        w = np.ones(n) / n
    else:
        w = w / w_sum
        
    # Ensure float64
    support = s.astype(np.float64)
    weights = w.astype(np.float64)
    
    return support, weights
