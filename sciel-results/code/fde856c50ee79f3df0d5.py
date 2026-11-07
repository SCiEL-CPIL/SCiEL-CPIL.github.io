import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Convert to numpy arrays for efficiency
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Basic validation
    if prizes.shape != (n_constraints, n_items) or weights.shape != (n_constraints, n_items):
        raise ValueError("Invalid shape")
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values")
    
    # Calculate total prize for each item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate relative weight for each item and constraint
    rel_weights = weights / capacity
    
    # Determine the bottleneck relative weight for each item
    bottleneck = np.max(rel_weights, axis=0)
    
    # Compute multiple heuristic components and combine them
    
    # 1. Prize efficiency relative to bottleneck constraint
    epsilon = 1e-9
    eff_bottleneck = total_prize / (bottleneck + epsilon)
    
    # 2. Prize density relative to total weight
    total_weight = np.sum(weights, axis=0)
    eff_total = total_prize / (total_weight + epsilon)
    
    # 3. Consider the minimum slack: how much room is left in the tightest constraint
    # Items that fit easily in all constraints are better
    min_slack = 1.0 - bottleneck  # fraction of tightest constraint remaining
    min_slack = np.clip(min_slack, epsilon, 1.0)
    
    # 4. Weighted combination of heuristics
    # Emphasize efficiency but also reward items that fit well
    heuristic = 0.6 * eff_bottleneck + 0.3 * eff_total + 0.1 * (total_prize * min_slack)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Optional: try a few alternative combinations if we have time
    # But given the simple nature, the above should be sufficient
    
    return heuristic
