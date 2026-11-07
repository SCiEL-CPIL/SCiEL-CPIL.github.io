"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a marginal utility heuristic
    that weights per-constraint efficiency by scarcity (top-k benchmark).
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger is more attractive)
    """
    # Validate inputs
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Input arrays must have shape (5, 100)")
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Input arrays must contain only finite values")
    
    n_constraints = 5
    n_items = 100
    capacity = 25.0
    
    # Compute per-constraint efficiency: prize / weight for each constraint
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    efficiency = np.zeros((n_constraints, n_items))
    for c in range(n_constraints):
        w = weights[c]
        p = prizes[c]
        # Where weight is very small, efficiency is high (good item for that constraint)
        efficiency[c] = np.where(w > eps, p / (w + eps), p / eps)
    
    # For each constraint, compute the top-k most efficient items as a benchmark
    # This represents the "scarcity" of that resource
    k = 10  # top-k items to consider for benchmark
    scarcity = np.zeros(n_constraints)
    for c in range(n_constraints):
        # Get top-k efficiency values for this constraint
        top_k_eff = np.sort(efficiency[c])[-k:]
        # Scarcity: higher when top items are very efficient (resource is "tight")
        # Use mean of top-k as benchmark
        scarcity[c] = np.mean(top_k_eff)
    
    # Compute marginal utility for each item
    # An item is valuable if it's efficient in constraints that are scarce
    # Weight each constraint's contribution by its scarcity
    marginal_utility = np.zeros(n_items)
    for c in range(n_constraints):
        # Contribution from constraint c: efficiency * scarcity
        # Items that are efficient in scarce constraints get higher scores
        marginal_utility += efficiency[c] * scarcity[c]
    
    # Normalize and ensure positivity
    # Add a small base to ensure all values are positive
    min_val = np.min(marginal_utility)
    if min_val < 0:
        marginal_utility = marginal_utility - min_val
    
    # Ensure all values are positive (as required by the task)
    marginal_utility = np.maximum(marginal_utility, 1e-9)
    
    return marginal_utility
# EVOLVE-BLOCK-END
