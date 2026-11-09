"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using LP relaxation dual prices.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint dimension
        weights: array of shape (5, 100) - weight values for each constraint dimension
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger is more attractive)
    """
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin before 300s limit
    
    n_constraints, n_items = prizes.shape
    
    # Solve LP relaxation to get dual prices (shadow prices)
    # Maximize sum of prizes subject to weight constraints
    # LP: max c^T x s.t. A x <= b, 0 <= x <= 1
    # Where c = prizes (flattened), A = weights, b = capacities
    
    # For multiple constraints, we want to maximize total prize
    # The "prize" for an item is typically the sum or some aggregate
    # Let's use the sum of prizes across all dimensions as the objective
    
    c = prizes.sum(axis=0)  # Objective: maximize sum of all prizes
    A_ub = weights  # Each row is a constraint: sum_j weights[i,j]*x_j <= capacity
    b_ub = np.full(n_constraints, 25.0)  # Capacity for each constraint
    
    bounds = [(0, 1) for _ in range(n_items)]
    
    try:
        # Use HiGHS solver for speed
        result = linprog(
            -c,  # Minimize -c^T x
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )
        
        if result.success and result.ineqlin is not None:
            # Dual prices (shadow prices) for the inequality constraints
            # result.ineqlin gives the dual variables for A_ub constraints
            dual_prices = result.ineqlin  # Shape: (n_constraints,)
        else:
            # Fallback: use simple ratio heuristic
            dual_prices = np.ones(n_constraints)
    except Exception:
        # Fallback: use simple ratio heuristic
        dual_prices = np.ones(n_constraints)
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback
        total_weights = weights.sum(axis=0)
        scores = prizes.sum(axis=0) / (total_weights + 1e-9)
        return np.maximum(scores, 1e-9)
    
    # Compute reduced cost for each item
    # Reduced cost = prize - sum(dual_price[i] * weight[i,j] for all constraints i)
    # This tells us how much "value" an item has above the shadow price of its resource usage
    
    # Weighted sum of weights by dual prices for each item
    weighted_weights = weights.T @ dual_prices  # Shape: (n_items,)
    
    # Prize for each item (sum across constraints)
    item_prizes = prizes.sum(axis=0)  # Shape: (n_items,)
    
    # Reduced cost: positive means item is "profitable" relative to dual prices
    reduced_costs = item_prizes - weighted_weights
    
    # The heuristic should favor items with high reduced cost
    # But we need to be careful: very negative reduced costs should get low scores
    
    # Use a smooth transformation to ensure positive values
    # Add a small offset to make all values positive, then use exponential or power
    
    # Find the minimum reduced cost to shift
    min_rc = np.min(reduced_costs)
    shifted_rc = reduced_costs - min_rc  # Now all >= 0
    
    # Use a power transformation to amplify differences
    # Higher power emphasizes items with much better reduced costs
    heuristic = shifted_rc ** 1.5
    
    # Ensure all values are positive (as required by the task)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline again
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
