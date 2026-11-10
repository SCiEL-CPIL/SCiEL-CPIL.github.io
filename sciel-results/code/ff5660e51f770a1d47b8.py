"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    Uses a Lagrangian relaxation approach where constraint multipliers are
    estimated from a greedy approximation of the knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # 1. Estimate constraint tightness (Lagrange multipliers) using a greedy fill
    # Sorting key: Total Prize / Max Weight (penalizes items that are heavy in any single constraint)
    total_prize = prizes.sum(axis=0)
    max_weight = weights.max(axis=0)
    eps = 1e-9
    greedy_scores = total_prize / (max_weight + eps)
    
    # Sort items by greedy score descending
    sorted_indices = np.argsort(-greedy_scores)
    
    # Simulate greedy packing
    used_capacity = np.zeros(n_constraints)
    for idx in sorted_indices:
        w = weights[:, idx]
        if np.all(used_capacity + w <= capacity):
            used_capacity += w
    
    # 2. Calculate Lagrange multipliers
    # Lambda is higher when remaining capacity is lower (scarcer resource)
    # lambda_j = 1 / (remaining_capacity_j + 1)
    remaining = capacity - used_capacity
    lambdas = 1.0 / (remaining + 1.0)
    
    # 3. Compute Lagrangian value for each item
    # Value = Total Prize - Sum(Lambda_j * Weight_ij)
    # Cost vector for each item
    costs = np.sum(lambdas * weights, axis=0)
    lagrangian_values = total_prize - costs
    
    # 4. Transform to positive heuristic scores
    # Shift so min value is 1.0 (avoids zeros/negatives which are clamped to 1e-9 in ACO)
    # This preserves relative differences while ensuring positivity
    min_val = lagrangian_values.min()
    heur = lagrangian_values - min_val + 1.0
    
    # Ensure finite and positive
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
