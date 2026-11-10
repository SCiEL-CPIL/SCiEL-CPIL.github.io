# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Vectorized computation:
    # P = sum_{k,l} w_k * w_l * F(2*s_l - s_k)
    # where F(t) = sum_{i,j: s_i+s_j < t} w_i * w_j
    #
    # For each pair (k,l), threshold = 2*s_l - s_k
    # We need the CDF of the pair-sum distribution at that threshold.
    
    # Pair sums: pair_sums[i,j] = s_i + s_j
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # For each (k,l) pair, compute threshold
    # thresholds[k,l] = 2*s_l - s_k
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find how many pair sums are strictly less than it
    # Use searchsorted on sorted pair sums
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k,l), find index of first pair sum >= threshold
    # idx[k,l] = number of pair sums strictly less than thresholds[k,l]
    idx = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get the cumulative probability at that index
    # inner[k,l] = sum of w_i*w_j for pairs with sum < threshold
    inner = np.zeros_like(thresholds)
    mask = idx > 0
    inner[mask] = cum_probs[idx[mask] - 1]
    
    # P = sum_{k,l} w_k * w_l * inner[k,l]
    wl = w_sorted[:, None] * w_sorted[None, :]
    P = np.sum(wl * inner)
    
    return P

def _neg_P(params, n_points):
    """Negative P for minimization. params: first n_points are weights (log-space), rest are support points."""
    log_w = params[:n_points]
    support = params[n_points:]
    
    # Convert log weights to weights
    w = np.exp(log_w - np.max(log_w))  # For numerical stability
    
    return -_compute_P(support, w)

def _simulated_annealing(initial_support, initial_weights, deadline, n_evals=None):
    """Simulated annealing to optimize support and weights jointly."""
    n = len(initial_support)
    
    def _encode(support, weights):
        log_w = np.log(weights + 1e-15)
        return np.concatenate([log_w, support])
    
    def _decode(params):
        log_w = params[:n]
        support = params[n:]
        w = np.exp(log_w - np.max(log_w))
        support = np.maximum(support, 0.0)
        return support, w
    
    def _score(params):
        support, w = _decode(params)
        return _compute_P(support, w)
    
    best_params = _encode(initial_support, initial_weights)
    best_score = _score(best_params)
    current_params = best_params.copy()
    current_score = best_score
    
    # Temperature schedule
    T0 = 0.5
    T_min = 1e-6
    cooling_rate = 0.995
    
    start_time = time.time()
    eval_count = 0
    
    while time.time() < deadline:
        T = max(T_min, T0 * (cooling_rate ** eval_count))
        
        # Generate perturbation
        perturbation_scale = 0.1 * (T / T0 + 0.1)
        
        new_params = current_params + np.random.randn(len(current_params)) * perturbation_scale
        
        # Ensure support non-negative
        new_params[n:] = np.maximum(new_params[n:], 0.0)
        
        new_score = _score(new_params)
        eval_count += 1
        
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            # Accept worse move with probability exp((new - current) / T)
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / max(T, 1e-10)):
                current_params = new_params
                current_score = new_score
        
        # Occasionally try larger jumps
        if eval_count % 100 == 0:
            jump_params = current_params + np.random.randn(len(current_params)) * 0.5
            jump_params[n:] = np.maximum(jump_params[n:], 0.0)
            jump_score = _score(jump_params)
            eval_count += 1
            if jump_score > current_score:
                current_params = jump_params
                current_score = jump_score
                if jump_score > best_score:
                    best_score = jump_score
                    best_params = jump_params.copy()
    
    support, w = _decode(best_params)
    return support, w, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Start with a reasonable initial distribution
    # Try different initial structures
    
    initial_configs = []
    
    # Config 1: Uniform on {0, 1, ..., n-1} with n=8
    n1 = 8
    support1 = np.arange(n1, dtype=np.float64)
    weights1 = np.ones(n1, dtype=np.float64) / n1
    initial_configs.append((support1, weights1))
    
    # Config 2: Skewed distribution - more mass at low values
    n2 = 8
    support2 = np.arange(n2, dtype=np.float64)
    weights2 = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.06], dtype=np.float64)
    weights2 = weights2 / weights2.sum()
    initial_configs.append((support2, weights2))
    
    # Config 3: Bimodal - mass at 0 and at higher values
    n3 = 6
    support3 = np.array([0, 1, 2, 4, 6, 8], dtype=np.float64)
    weights3 = np.array([0.25, 0.15, 0.1, 0.2, 0.15, 0.15], dtype=np.float64)
    weights3 = weights3 / weights3.sum()
    initial_configs.append((support3, weights3))
    
    # Config 4: Exponential-like decay
    n4 = 10
    support4 = np.arange(n4, dtype=np.float64)
    weights4 = np.exp(-0.3 * support4)
    weights4 = weights4 / weights4.sum()
    initial_configs.append((support4, weights4))
    
    # Config 5: Mass concentrated at 0 and 1
    n5 = 4
    support5 = np.array([0, 1, 2, 4], dtype=np.float64)
    weights5 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    weights5 = weights5 / weights5.sum()
    initial_configs.append((support5, weights5))
    
    # Config 6: Two-point distribution
    n6 = 2
    support6 = np.array([0.0, 1.0], dtype=np.float64)
    weights6 = np.array([0.5, 0.5], dtype=np.float64)
    initial_configs.append((support6, weights6))
    
    # Config 7: Three-point distribution
    n7 = 3
    support7 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    weights7 = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    initial_configs.append((support7, weights7))
    
    # Config 8: Geometric spacing
    n8 = 8
    support8 = np.array([0, 0.5, 1, 2, 4, 8, 16, 32], dtype=np.float64)
    weights8 = np.exp(-0.5 * np.arange(n8))
    weights8 = weights8 / weights8.sum()
    initial_configs.append((support8, weights8))
    
    # Evaluate initial configs
    for support, weights in initial_configs:
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Use simulated annealing for global search from each initial config
    for support, weights in initial_configs:
        if time.time() > deadline - 30:
            break
        
        try:
            opt_support, opt_weights, P = _simulated_annealing(support, weights, deadline)
            if P > best_P:
                best_P = P
                best_support = opt_support.copy()
                best_weights = opt_weights.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Clean up: ensure no NaN/inf, non-negative
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights is positive
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
