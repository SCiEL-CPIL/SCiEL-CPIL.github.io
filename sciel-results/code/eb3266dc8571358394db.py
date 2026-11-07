import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    epsilon = 1e-9

    # Safe weights to avoid division by zero
    safe_weights = np.where(weights > epsilon, weights, 1.0)  # (5, 100)

    # Per-constraint ratios: prizes[k, j] / weights[k, j]
    ratios = prizes / safe_weights  # (5, 100)

    # 1. Soft-Min-Ratio (Bottleneck) via log-sum-exp
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # As t -> 0, this approaches min(x)
    # Use a small temperature for close approximation to min
    t_soft = 0.1
    neg_ratios = -ratios / t_soft
    neg_ratios = np.clip(neg_ratios, -100, 100)  # Avoid overflow
    log_sum_exp = np.log(np.sum(np.exp(neg_ratios), axis=0))
    M = -t_soft * log_sum_exp  # (100,)

    # 2. Harmonic Mean of ratios for each item
    # H_j = 5 / sum(1/r_kj) where 1/r_kj = w_kj / p_kj
    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom  # (5, 100)
    sum_inv_ratios = np.sum(inv_ratios, axis=0)  # (100,)
    H = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)

    # 3. Geometric Blending: H^1.5 * M^0.5
    # Strong weight on overall efficiency (harmonic mean), moderate on bottleneck
    H_safe = np.maximum(H, epsilon)
    M_safe = np.maximum(M, epsilon)

    alpha = 1.5
    beta = 0.5

    heuristic = np.power(H_safe, alpha) * np.power(M_safe, beta)

    # Safety: replace non-finite or non-positive with epsilon
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    # Normalize to mean 1 to match pheromone scale
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones(100, dtype=np.float64)

    # Final safety check
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)

    if time.time() > deadline:
        return heuristic

    return heuristic
