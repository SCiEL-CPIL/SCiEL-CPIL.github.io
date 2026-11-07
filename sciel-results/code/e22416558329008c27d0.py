import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Calculate efficiency ratios for each constraint and item
    # R[k, j] = prize[k, j] / weight[k, j]
    # Use epsilon to avoid division by zero
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    ratios = prizes / safe_weights  # Shape (5, 100)

    # 1. Harmonic Mean of ratios for each item
    # H_j = n / sum(1/r_kj)
    # 1/r_kj = safe_weights[k, j] / prizes[k, j]
    # We need to handle prizes == 0. If prize is 0, ratio is 0, 1/ratio is inf.
    # If prize is 0, the item is useless. We want H_j to be 0.
    # So, if any prize is 0 (or very small), H_j should be 0.

    # Let's compute inverse ratios safely
    # If prizes[k,j] is 0, inverse ratio is inf.
    # Sum of inverses will be inf. H_j = 5/inf = 0.

    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom

    sum_inv_ratios = np.sum(inv_ratios, axis=0) # (100,)

    # Harmonic Mean H_j
    # If sum_inv_ratios is 0, it means all inv_ratios were 0, which implies all weights were 0 (and prizes > 0).
    # In that case, efficiency is infinite. We cap it or set to a large value.
    H = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)

    # 2. Soft-Min-Ratio (Bottleneck) M_j
    # Use soft-min via log-sum-exp for smoothness
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # As t -> 0, this approaches min(x)
    # Use a small temperature for close approximation to min
    t_soft = 0.05  # Smaller temperature for sharper bottleneck approximation
    neg_ratios = -ratios / t_soft
    # Clip to avoid overflow in exp
    neg_ratios = np.clip(neg_ratios, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(neg_ratios), axis=0))
    M = -t_soft * log_sum_exp  # (100,)

    # 3. Additional signal: total prize sum
    # Items with higher total prize are generally more attractive
    total_prize = np.sum(prizes, axis=0)  # (100,)
    total_prize_safe = np.maximum(total_prize, epsilon)

    # 4. Geometric Blending with three components
    # eta = H^alpha * M^beta * total_prize^gamma
    # H captures overall efficiency
    # M captures bottleneck (weakest constraint)
    # total_prize captures absolute value
    
    H_safe = np.maximum(H, epsilon)
    M_safe = np.maximum(M, epsilon)
    
    # Exponents tuned to balance the three signals
    alpha = 1.2
    beta = 0.6
    gamma = 0.3
    
    heuristic = np.power(H_safe, alpha) * np.power(M_safe, beta) * np.power(total_prize_safe, gamma)

    # 5. Normalization
    # Normalize to mean 1 to match pheromone scale
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # 6. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
