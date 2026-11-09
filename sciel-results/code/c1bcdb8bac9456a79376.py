# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,)
    Returns probability.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        return 0.0
    w = w / w_sum
    n = len(s)
    if n == 0:
        return 0.0
    
    # Compute all pairwise sums and their probabilities
    # We need sum_{i,j} w_i w_j I(s_i + s_j < T) for various T.
    # T = 2*s_l - s_k.
    # There are n^2 such thresholds.
    # For each threshold, we sum w_i w_j.
    # This is O(n^4) if done naively.
    # For small n (<= 10), O(n^4) is fine (10^4 = 10000 ops).
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum w_i * w_j for i, j such that s_i + s_j < threshold
            # Vectorized inner loop
            # s_i + s_j < threshold
            # Create matrix of sums
            # sums = s[:, None] + s[None, :]
            # mask = sums < threshold
            # inner = np.sum(w[:, None] * w[None, :] * mask)
            # This creates n x n arrays.
            
            # More efficient for small n:
            # Since n is small, just loop or use broadcasting
            sums = s[:, None] + s[None, :]
            mask = sums < threshold
            inner_sum = np.sum(w[:, None] * w[None, :] * mask)
            P += w[k] * w[l] * inner_sum
            
    return P

def _objective(params, n):
    """
    Objective function for minimizer (we minimize -P).
    params: array of length 2*n.
    First n are supports, next n are weights.
    """
    supports = params[:n]
    weights = params[n:]
    
    # Ensure weights are non-negative and sum to 1?
    # SLSQP will handle constraints.
    # But for safety in calculation, if any weight < 0, return large value.
    if np.any(weights < -1e-9) or np.any(supports < -1e-9):
        return 1e6
        
    p = _calc_P(supports, weights)
    return -p

def _optimize_measure(n, seed=0):
    """
    Optimize a discrete measure with n points.
    """
    np.random.seed(seed)
    
    # Initial guess
    # Start with a 2-point-like distribution spread over n points
    # e.g., mass at 0 and 1, with some noise
    supports = np.zeros(n)
    supports[-1] = 1.0
    # Add some intermediate points
    if n > 2:
        supports[1:-1] = np.linspace(0.1, 0.9, n-2)
        
    weights = np.zeros(n)
    weights[0] = 0.6
    weights[-1] = 0.4
    # Add small noise to other weights to break symmetry
    if n > 2:
        noise = np.random.rand(n-2) * 0.01
        weights[1:-1] = noise
        # Renormalize
        weights = weights / np.sum(weights)

    x0 = np.concatenate([supports, weights])
    
    # Constraints
    # Sum of weights = 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[n:]) - 1.0})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, None)) # supports >= 0
    for i in range(n):
        bounds.append((0.0, 1.0)) # weights in [0, 1]
        
    # Optimize
    try:
        res = minimize(_objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            best_supports = res.x[:n]
            best_weights = res.x[n:]
            # Clean up tiny negatives
            best_weights = np.maximum(best_weights, 0.0)
            best_weights = best_weights / np.sum(best_weights)
            return best_supports, best_weights, _calc_P(best_supports, best_weights)
        else:
            return None, None, -np.inf
    except Exception:
        return None, None, -np.inf

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    """
    start_time = time.time()
    deadline = 550.0 # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -np.inf
    
    # Try different n values
    # n=2 is the baseline we analyzed (~0.26)
    # n=3, 4, 5 might be better.
    n_values = [2, 3, 4, 5, 6]
    
    for n in n_values:
        if time.time() > deadline:
            break
            
        # Run a few seeds for robustness
        for seed in range(5):
            if time.time() > deadline:
                break
                
            sup, w, score = _optimize_measure(n, seed)
            if sup is not None and score > best_score:
                best_score = score
                best_support = sup
                best_weights = w
                
    # Fallback to seed program if optimization fails
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure valid output format
        best_support = np.asarray(best_support, dtype=np.float64)
        best_weights = np.asarray(best_weights, dtype=np.float64)
        # Flatten
        best_support = best_support.flatten()
        best_weights = best_weights.flatten()
        
        # Final check for validity
        if len(best_support) != len(best_weights) or len(best_support) < 1:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
            
        if np.any(np.isnan(best_support)) or np.any(np.isnan(best_weights)):
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
            
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
            
        if np.sum(best_weights) <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights

# EVOLVE-BLOCK-END
