import numpy as np
import time
from scipy.optimize import differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized O(n^2 log n) method.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 0.0
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return 0.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    
    p = weights / w_sum
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Sort for prefix-sum queries
    sort_idx = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    total_prob = 0.0
    
    for k in range(n):
        threshold = 2.0 * support[k]
        T = threshold - support  # For each l, we need S < T_l
        
        indices = np.searchsorted(S_sorted, T, side='left')
        sums_for_l = W_cumsum_padded[indices]
        
        prob3 = np.dot(p, sums_for_l)
        total_prob += p[k] * prob3
    
    return total_prob


def _params_to_measure(params, n_points):
    """
    Convert raw optimization parameters to (support, weights).
    params layout:
      - First n_points-1 values: log-spaced support positions (exp of param)
      - Last n_points values: logit-transformed weights (softmax)
    Support[0] is fixed at 0.0 for structural advantage.
    """
    # Support: first point at 0, rest from log-space
    log_supports = params[:n_points - 1]
    support = np.zeros(n_points)
    support[0] = 0.0
    support[1:] = np.exp(log_supports)
    support = np.sort(support)
    
    # Weights: softmax of last n_points params
    w_params = params[n_points - 1:]
    # Shift for numerical stability
    w_shifted = w_params - np.max(w_params)
    weights = np.exp(w_shifted)
    weights = weights / np.sum(weights)
    
    return support, weights


def _neg_prob(params, n_points):
    """Negative probability for minimization."""
    support, weights = _params_to_measure(params, n_points)
    
    # Basic validity
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
    if np.sum(weights) <= 1e-9:
        return 0.0
    
    prob = _calc_prob_fast(support, weights)
    return -prob


def beat_the_average_measure() -> tuple:
    """
    Global optimization via differential evolution over compact parameter space.
    Key structural insight: fixing one support point at 0 and using log-space
    for other supports + softmax for weights gives a smooth, bounded parameterization
    amenable to global search.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s kill
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # Try multiple sizes: 2, 3, 4, 5 points
    # For each size, run differential evolution
    for n_points in [2, 3, 4, 5]:
        if time.time() > deadline - 60.0:
            break
        
        # Parameter bounds
        # log_supports: exp(log_s) should be in [0.01, 100] -> log_s in [log(0.01), log(100)]
        log_min = np.log(0.01)
        log_max = np.log(100.0)
        
        # Weights logits: [-5, 5] covers a wide range of weight ratios
        w_min = -5.0
        w_max = 5.0
        
        # Build bounds
        bounds = []
        for _ in range(n_points - 1):
            bounds.append((log_min, log_max))
        for _ in range(n_points):
            bounds.append((w_min, w_max))
        
        # Initial best for this size
        size_best_prob = -1.0
        size_best_support = None
        size_best_weights = None
        
        # Run differential evolution
        # Use a reasonable number of iterations; DE is expensive per eval
        # but global. Limit maxiter based on time budget.
        maxiter = 50
        popsize = 10
        
        try:
            result = differential_evolution(
                _neg_prob,
                bounds=bounds,
                args=(n_points,),
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                updating='deferred',
                x0=None
            )
        except Exception:
            # Fallback: try a simpler manual DE-like approach
            pass
        else:
            if result.success or result.fun < 0:
                prob = -result.fun
                if prob > size_best_prob:
                    size_best_prob = prob
                    size_best_support, size_best_weights = _params_to_measure(
                        result.x, n_points
                    )
        
        # Also try a few random initializations with local polish
        # to complement DE (DE may miss narrow basins)
        n_random_starts = 20
        for rs in range(n_random_starts):
            if time.time() > deadline - 30.0:
                break
            
            # Random params
            log_s_rand = np.random.uniform(log_min, log_max, size=n_points - 1)
            w_rand = np.random.uniform(w_min, w_max, size=n_points)
            params_rand = np.concatenate([log_s_rand, w_rand])
            
            support, weights = _params_to_measure(params_rand, n_points)
            prob = _calc_prob_fast(support, weights)
            
            if prob > size_best_prob:
                size_best_prob = prob
                size_best_support = support.copy()
                size_best_weights = weights.copy()
        
        # Update global best
        if size_best_prob > best_prob and size_best_support is not None:
            best_prob = size_best_prob
            best_support = size_best_support.copy()
            best_weights = size_best_weights.copy()
    
    # Fallback: if nothing found, use a known decent distribution
    if best_support is None or best_prob < 0:
        # Try a few hand-crafted distributions
        candidates = [
            (np.array([0.0, 1.0]), np.array([0.5, 0.5])),
            (np.array([0.0, 0.5, 1.0]), np.array([0.4, 0.3, 0.3])),
            (np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.4, 0.3])),
            (np.array([0.0, 0.25, 0.5, 1.0]), np.array([0.3, 0.25, 0.25, 0.2])),
        ]
        for s, w in candidates:
            p = _calc_prob_fast(s, w)
            if p > best_prob:
                best_prob = p
                best_support = s.copy()
                best_weights = w.copy()
    
    # Final validation and cleanup
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        # Ensure validity
        if np.any(np.isnan(best_support)) or np.any(np.isnan(best_weights)):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
        if np.any(np.isinf(best_support)) or np.any(np.isinf(best_weights)):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
        if np.any(best_support < -1e-9):
            best_support = np.maximum(best_support, 0.0)
        if np.any(best_weights < -1e-9):
            best_weights = np.maximum(best_weights, 0.0)
        
        w_sum = np.sum(best_weights)
        if w_sum <= 1e-9:
            best_weights = np.ones(len(best_support)) / len(best_support)
        else:
            best_weights = best_weights / w_sum
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
