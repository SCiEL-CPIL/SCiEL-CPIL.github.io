import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses Pareto-dominance filter with soft-max weight normalization.
    Non-dominated items get boosted scores; dominated items are penalized.
    """
    start_time = time.time()
    deadline = 325.0
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Soft-max of weights per item (smoother than hard max)
    # Use log-sum-exp with temperature for smoothness
    # soft_max = log(sum(exp(w/T))) * T, but for T->0 this approaches max
    # For T=0.1, we get a smooth approximation
    T = 0.1
    w_scaled = weights / T
    w_max = np.max(w_scaled, axis=0, keepdims=True)
    log_sum_exp = w_max + np.log(np.sum(np.exp(w_scaled - w_max), axis=0, keepdims=True))
    soft_max_weights = log_sum_exp * T  # shape (1, n_items)
    soft_max_weights = soft_max_weights[0]  # shape (n_items,)
    
    # Base heuristic: prize / soft_max_weight
    base_heuristic = prizes / (soft_max_weights + 1e-9)
    
    # Pareto dominance check
    is_dominated = np.zeros(n_items, dtype=bool)
    domination_strength = np.zeros(n_items, dtype=float)
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            if prizes[j] >= prizes[i]:
                if np.all(weights[j, :] <= weights[i, :]):
                    is_dominated[i] = True
                    dom_strength = 1.0 - (prizes[i] / (prizes[j] + 1e-9))
                    domination_strength[i] = max(domination_strength[i], dom_strength)
    
    # Apply Pareto filter
    boost = np.ones(n_items, dtype=float)
    for i in range(n_items):
        if is_dominated[i]:
            boost[i] = 1.0 - domination_strength[i] * 0.5
        else:
            boost[i] = 1.5
    
    heuristic = base_heuristic * boost
    
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    if time.time() - start_time > deadline:
        return np.ones(100)
        
    return heuristic
