# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Pairwise sums and probabilities
    # pair_sums[k, l] = s_k + s_l
    # pair_probs[k, l] = p_k * p_l
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Sort pair sums to use CDF for efficient threshold queries
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Thresholds: T[k, l] = 2*s_l - s_k
    # We want sum_{i,j} p_i p_j [s_i + s_j < T[k,l]]
    # This is CDF_S(T[k,l]) where CDF_S(t) = sum_{s_i+s_j < t} p_i p_j
    
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # searchsorted with side='left' gives the first index where sorted_sums >= T
    # So the number of elements < T is that index.
    # The sum of probs for those elements is cum_probs[idx - 1] if idx > 0, else 0.
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # P = sum_{k,l} p_k p_l * CDF_S(2*s_l - s_k)
    prob_kl = p[:, None] * p[None, :]
    P = np.sum(prob_kl * cdf_vals)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 590  # Keep a 10s margin from the 600s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    # Strategy 1: Two-point distributions
    # Support [0, 1], weights [1-p, p]
    # P = p(1-p)^2(1+2p)
    # Maximize this analytically or via grid.
    # Derivative of g(p) = p(1-p)^2(1+2p)
    # g(p) = p(1 - 2p + p^2)(1+2p) = p(1 + 2p - 2p - 4p^2 + p^2 + 2p^3) = p(1 - 3p^2 + 2p^3) = p - 3p^3 + 2p^4
    # g'(p) = 1 - 9p^2 + 8p^3
    # 8p^3 - 9p^2 + 1 = 0
    # (p-1)(8p^2 - p - 1) = 0
    # p=1 is a root.
    # 8p^2 - p - 1 = 0 -> p = (1 +/- sqrt(1 + 32))/16 = (1 +/- sqrt(33))/16
    # sqrt(33) approx 5.744
    # p = (1 + 5.744)/16 approx 6.744/16 approx 0.4215
    # Let's check p = 0.4215
    p_opt = (1 + np.sqrt(33)) / 16.0
    support2 = np.array([0.0, 1.0])
    weights2 = np.array([1.0 - p_opt, p_opt])
    score2 = _evaluate_measure(support2, weights2)
    update_best(support2, weights2, score2)
    
    # Also check grid just in case of numerical issues or my math error
    for p in np.linspace(0.1, 0.9, 100):
        if time_exceeded(): break
        w = np.array([1-p, p])
        s = _evaluate_measure(support2, w)
        update_best(support2, w, s)

    # Strategy 2: Three-point distributions
    # Support [0, 1, x]
    # This allows more flexibility.
    # Let's try a grid search over x and weights.
    # Weights w0, w1, w2.
    # To keep it simple, let's fix support [0, 1, 2] and optimize weights.
    # Or support [0, 1, x] with x in [1, 10].
    
    # Let's try support [0, 1, 2] first.
    # Random search for weights on [0, 1, 2]
    np.random.seed(42)
    
    # Simulated Annealing or Hill Climbing for weights on fixed support
    def optimize_weights_fixed_support(support, start_weights, iters=2000, step_size=0.05):
        w = start_weights.copy()
        w = w / np.sum(w)
        best_w = w.copy()
        best_s = _evaluate_measure(support, w)
        
        for i in range(iters):
            if time_exceeded():
                break
            
            # Perturb
            w_new = w + np.random.randn(len(w)) * step_size
            w_new = np.maximum(w_new, 0)
            sum_w = np.sum(w_new)
            if sum_w < 1e-9:
                continue
            w_new = w_new / sum_w
            
            s_new = _evaluate_measure(support, w_new)
            
            if s_new > best_s:
                best_s = s_new
                best_w = w_new.copy()
                w = w_new
            else:
                # Accept with some probability? Or just reject.
                # Simple hill climbing: reject.
                pass
        
        return best_w, best_s

    # Try several support configurations
    support_configs = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.5]),
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 1.0, 4.0]),
        np.array([0.0, 1.0, 5.0]),
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.1, 1.0]),
        np.array([0.0, 0.01, 1.0]),
    ]
    
    for sup in support_configs:
        if time_exceeded(): break
        # Try a few random starts
        for _ in range(5):
            if time_exceeded(): break
            w_start = np.random.dirichlet(np.ones(3))
            w_opt, s_opt = optimize_weights_fixed_support(sup, w_start, iters=500, step_size=0.1)
            update_best(sup, w_opt, s_opt)
            
            # Also try starting from uniform
            w_start = np.ones(3) / 3.0
            w_opt, s_opt = optimize_weights_fixed_support(sup, w_start, iters=500, step_size=0.1)
            update_best(sup, w_opt, s_opt)

    # Strategy 3: Four-point distributions
    # Support [0, 1, 2, 3]
    sup4 = np.array([0.0, 1.0, 2.0, 3.0])
    for _ in range(5):
        if time_exceeded(): break
        w_start = np.random.dirichlet(np.ones(4))
        w_opt, s_opt = optimize_weights_fixed_support(sup4, w_start, iters=1000, step_size=0.05)
        update_best(sup4, w_opt, s_opt)
    
    sup4b = np.array([0.0, 1.0, 2.0, 5.0])
    for _ in range(5):
        if time_exceeded(): break
        w_start = np.random.dirichlet(np.ones(4))
        w_opt, s_opt = optimize_weights_fixed_support(sup4b, w_start, iters=1000, step_size=0.05)
        update_best(sup4b, w_opt, s_opt)

    # Strategy 4: Random support and weights for small n
    # n=3, 4, 5
    for n in [3, 4, 5]:
        if time_exceeded(): break
        for _ in range(20):
            if time_exceeded(): break
            # Random support in [0, 10]
            sup = np.sort(np.random.rand(n) * 10.0)
            # Ensure distinct? Not strictly required but helps.
            # Random weights
            w = np.random.dirichlet(np.ones(n))
            w_opt, s_opt = optimize_weights_fixed_support(sup, w, iters=500, step_size=0.05)
            update_best(sup, w_opt, s_opt)

    # Strategy 5: Geometric-like distributions on integer support
    # w_i = q^i
    # We found good scores with small q in the previous run?
    # Let's refine the search for geometric on {0, 1, ..., n-1}
    for n in [3, 4, 5, 6]:
        if time_exceeded(): break
        for q in np.linspace(0.1, 0.9, 20):
            if time_exceeded(): break
            sup = np.arange(n, dtype=np.float64)
            w = np.power(q, sup)
            s = _evaluate_measure(sup, w)
            update_best(sup, w, s)
            
            # Also try optimizing weights starting from this geometric
            w_opt, s_opt = optimize_weights_fixed_support(sup, w, iters=200, step_size=0.05)
            update_best(sup, w_opt, s_opt)

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])

    return best_support, best_weights

# EVOLVE-BLOCK-END
