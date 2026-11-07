import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints = prizes.shape[0]  # 5
    n_items = prizes.shape[1]        # 100
    
    epsilon = 1e-9
    
    # Safe division: avoid division by zero
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    safe_prizes = np.where(prizes > epsilon, prizes, epsilon)
    
    # Per-constraint efficiency ratios: R[k, j] = prize[k,j] / weight[k,j]
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    
    # 1. Harmonic Mean of ratios for each item
    # H_j = n / sum(1/r_kj)
    # 1/r_kj = weight[k,j] / prize[k,j]
    inv_ratios = safe_weights / safe_prizes  # (5, 100)
    sum_inv_ratios = np.sum(inv_ratios, axis=0)  # (100,)
    
    # If sum_inv_ratios is very small, harmonic mean is large
    H = np.where(sum_inv_ratios > epsilon, 
                 float(n_constraints) / np.maximum(sum_inv_ratios, epsilon), 
                 1e6)
    
    # 2. Soft-min of ratios (smooth approximation to min)
    # soft_min(x) = -logsumexp(-x)
    # This is smooth and approximates min when temperature is low
    # Use a temperature parameter to control smoothness
    temp = 0.1  # Lower = closer to hard min
    neg_ratios = -ratios  # (5, 100)
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)  # (1, 100)
    # For numerical stability
    shifted = neg_ratios - max_neg  # (5, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted / temp), axis=0)) + max_neg[0]
    soft_min = -log_sum_exp  # (100,)
    
    # 3. Also compute a "prize density" - total prize per total weight
    total_prize = np.sum(safe_prizes, axis=0)  # (100,)
    total_weight = np.sum(safe_weights, axis=0)  # (100,)
    prize_density = total_prize / np.maximum(total_weight, epsilon)  # (100,)
    
    # 4. Geometric blending with adjusted exponents
    # Blend: H^alpha * soft_min^beta * prize_density^gamma
    # Normalize each component to have mean 1 before blending for stability
    
    H_safe = np.maximum(H, epsilon)
    sm_safe = np.maximum(soft_min, epsilon)
    pd_safe = np.maximum(prize_density, epsilon)
    
    # Normalize to mean 1
    H_norm = H_safe / np.mean(H_safe)
    sm_norm = sm_safe / np.mean(sm_safe)
    pd_norm = pd_safe / np.mean(pd_safe)
    
    # Exponents: favor harmonic mean and soft-min, less on prize density
    alpha = 0.5
    beta = 0.35
    gamma = 0.15
    
    heuristic = np.power(H_norm, alpha) * np.power(sm_norm, beta) * np.power(pd_norm, gamma)
    
    # 5. Final normalization to mean 1
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones(n_items)
        
    # 6. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (n_items,):
        heuristic = heuristic.reshape(n_items)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
