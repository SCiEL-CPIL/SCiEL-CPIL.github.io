"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_heuristics(prizes, weights, deadline=None):
    """
    Compute heuristic attractiveness scores using Lagrangian relaxation.
    Estimates dual prices via subgradient descent, then scores items by
    their reduced cost (prize - lambda dot weights).
    """
    start_time = time.time()
    
    # Validate inputs
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.ndim != 2 or weights.ndim != 2:
        raise ValueError("prizes and weights must be 2D arrays")
    
    if prizes.shape[0] != weights.shape[0]:
        raise ValueError("prizes and weights must have same number of constraints")
    
    if prizes.shape[1] != weights.shape[1]:
        raise ValueError("prizes and weights must have same number of items")
    
    num_constraints, n_items = prizes.shape
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("prizes and weights must contain only finite values")
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (n_items,)
    
    # Capacity per constraint
    capacity = np.full(num_constraints, 25.0)
    
    # Lagrangian dual: minimize g(lambda) = sum_j max(0, total_prize[j] - lambda @ weights[:,j]) + lambda @ capacity
    # over lambda >= 0
    # Subgradient: for each constraint c, subgradient_c = capacity[c] - sum_j {weights[c,j] if item j is "selected"}
    # where item j is selected if total_prize[j] - lambda @ weights[:,j] > 0
    
    # Initialize lambda
    lam = np.zeros(num_constraints)
    
    # Subgradient descent with adaptive step size
    # Use Polyak step: step_k = (g(lam_k) - g*) / ||subgrad||^2, but we don't know g*
    # Use a simple adaptive step instead
    
    num_iters = 200
    base_step = 0.5
    
    best_lam = lam.copy()
    best_g = np.inf
    
    for it in range(num_iters):
        # Check deadline
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute reduced costs for all items
        # reduced_cost[j] = total_prize[j] - lam @ weights[:,j]
        reduced_costs = total_prize - weights.T @ lam  # shape (n_items,)
        
        # Items "selected" in Lagrangian relaxation
        selected = reduced_costs > 0
        
        # Dual function value
        g_val = np.sum(np.maximum(reduced_costs, 0.0)) + lam @ capacity
        
        # Track best
        if g_val < best_g:
            best_g = g_val
            best_lam = lam.copy()
        
        # Subgradient: subgrad_c = capacity[c] - sum_j selected[j] * weights[c,j]
        if np.any(selected):
            used = weights[:, selected].sum(axis=1)  # shape (num_constraints,)
        else:
            used = np.zeros(num_constraints)
        
        subgrad = capacity - used
        
        # Step size: use a diminishing step
        step = base_step / (1.0 + 0.05 * it)
        
        # Update lambda (project to non-negative)
        lam = np.maximum(lam + step * subgrad, 0.0)
    
    # Use the best lambda found
    lam = best_lam
    
    # Final reduced costs
    reduced_costs = total_prize - weights.T @ lam
    
    # The heuristic should be proportional to reduced cost, but ensure positivity
    # Items with negative reduced cost are "expensive" relative to their prize
    # We want larger = more attractive, so use max(reduced_cost, small_positive)
    # But also consider: items with very high reduced cost are great
    
    # Use a smooth transformation to ensure positivity and reasonable scale
    # heuristic = max(reduced_cost, 0) + small_base
    # Or use exp to amplify differences: heuristic = exp(reduced_cost / scale)
    
    # Let's use: heuristic = reduced_cost + offset, where offset ensures positivity
    # The offset should be such that items with negative reduced cost still get a small positive value
    min_rc = np.min(reduced_costs)
    offset = -min_rc + 1e-6 if min_rc < 0 else 1e-6
    
    heuristic = reduced_costs + offset
    
    # Ensure all positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in ant colony optimization
    of the multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (num_constraints, n_items) with values in [0,1]
        weights: array of shape (num_constraints, n_items) with values in [0,1]
    
    Returns:
        array of shape (n_items,) with one heuristic score per item (larger = more attractive)
    """
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, 10s safety margin
    
    try:
        result = _compute_heuristics(prizes, weights, deadline)
        
        # Validate output
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        # Ensure shape matches number of items
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception as e:
        # Fallback: return a simple valid heuristic
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
        
        # Simple fallback: prize/weight ratio
        total_prize = prizes_arr.sum(axis=0)
        sum_weights = weights_arr.sum(axis=0)
        fallback = total_prize / (sum_weights + 1e-9)
        fallback = np.maximum(fallback, 1e-9)
        return fallback
# EVOLVE-BLOCK-END
