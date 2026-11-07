import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization using a competitive
    exclusion score based on relative efficiency.
    
    The heuristic for item i is the soft-min over constraints c of the ratio
    of item i's efficiency in c to the average efficiency of all items in c.
    This penalizes items that are inefficient in any constraint relative to
    the competition in that constraint.
    """
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s limit
    
    # Ensure inputs are numpy arrays of correct shape
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != weights.shape:
        raise ValueError("prizes and weights must have the same shape")
        
    if len(prizes.shape) != 2:
        raise ValueError("prizes and weights must be 2D")
        
    C, N = prizes.shape
    
    # Safety check for time
    if time.time() > deadline:
        return np.ones(N) * 1e-9

    epsilon = 1e-9
    
    # Safe weights to avoid division by zero
    safe_weights = np.where(weights > epsilon, weights, epsilon)
    
    # 1. Calculate total weights and total prizes per constraint
    # W_c = sum of weights in constraint c
    # P_c = sum of prizes in constraint c
    W_c = np.sum(safe_weights, axis=1) # shape (C,)
    P_c = np.sum(prizes, axis=1)       # shape (C,)
    
    # Avoid division by zero in average efficiency
    safe_W_c = np.where(W_c > epsilon, W_c, epsilon)
    safe_P_c = np.where(P_c > epsilon, P_c, epsilon)
    
    # 2. Calculate average efficiency per constraint
    # avg_eff_c = P_c / W_c
    avg_eff = safe_P_c / safe_W_c # shape (C,)
    
    # 3. Calculate item efficiency per constraint
    # eff[i, c] = prizes[i, c] / weights[i, c]
    eff = prizes / safe_weights # shape (C, N)
    
    # 4. Calculate relative efficiency
    # rel_eff[i, c] = eff[i, c] / avg_eff[c]
    # This normalizes the efficiency by the "market price" of capacity in that constraint.
    rel_eff = eff / avg_eff[:, np.newaxis] # shape (C, N)
    
    # Safety check for time
    if time.time() > deadline:
        return np.ones(N) * 1e-9

    # 5. Soft-min over constraints
    # We want to highlight items that are good in ALL constraints.
    # A hard min would be: min_c(rel_eff[i, c])
    # Soft-min: -1/alpha * log(sum(exp(-alpha * rel_eff)))
    
    # Choose alpha. 
    # If alpha is small, it's like a geometric mean (product).
    # If alpha is large, it's like a hard min.
    # The previous program used alpha=100 for raw ratios.
    # Relative efficiencies are centered around 1.
    # Let's try alpha=50.0.
    alpha = 50.0
    
    # For numerical stability, subtract max before exp
    # We are computing soft-min, so we look at -rel_eff.
    # max(-rel_eff) = -min(rel_eff)
    # Let's compute log-sum-exp of -alpha * rel_eff
    
    neg_alpha_rel_eff = -alpha * rel_eff
    
    # max_neg_alpha_rel_eff = max over constraints (axis=0)
    max_neg_alpha_rel_eff = np.max(neg_alpha_rel_eff, axis=0, keepdims=True)
    
    # exp_terms = exp(neg_alpha_rel_eff - max_neg_alpha_rel_eff)
    # Argument is <= 0, so exp is <= 1, safe from overflow.
    exp_terms = np.exp(neg_alpha_rel_eff - max_neg_alpha_rel_eff)
    
    sum_exp = np.sum(exp_terms, axis=0)
    
    # soft_min = -1/alpha * (max_neg_alpha_rel_eff + log(sum_exp))
    # Note: max_neg_alpha_rel_eff is shape (1, N)
    log_sum = np.log(sum_exp)
    
    soft_min_log = -1.0 / alpha * (max_neg_alpha_rel_eff[0] + log_sum)
    
    heuristic = np.exp(soft_min_log)
    
    # Ensure all values are positive and finite
    # If any value is non-finite or <= 0, replace with 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Final time check
    if time.time() > deadline:
        return np.ones(N) * 1e-9
        
    return heuristic
