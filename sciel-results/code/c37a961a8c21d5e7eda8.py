"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Rank-based constraint-aware heuristic.
    1. For each item, determine its 'binding constraint' (the constraint where it uses the most capacity fraction).
    2. Compute a base score as prize / max_weight (rewarding high prize, low max weight).
    3. Penalize items based on how 'crowded' their binding constraint is by other high-prize items.
       Specifically, for each constraint c, compute the sum of prizes of items that bind to c.
       Items binding to a heavily loaded constraint get a lower score.
    4. Combine with a power amplification to sharpen the distribution.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure 1D arrays
    if prizes.ndim == 2:
        # If prizes is (5, 100), sum across constraints to get total prize per item
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Capacity is 25 for each constraint
    capacity = 25.0
    
    # Step 1: For each item, find the binding constraint
    # weight_fraction[i, j] = weights[j, i] / capacity
    weight_fractions = weights / capacity  # shape (n_constraints, n_items)
    
    # The binding constraint for item i is argmax_j weight_fractions[j, i]
    binding_constraint = np.argmax(weight_fractions, axis=0)  # shape (n_items,)
    
    # Step 2: Base score = prize / max_weight
    max_weight = weights.max(axis=0)  # shape (n_items,)
    base_score = prizes / (max_weight + 1e-9)  # shape (n_items,)
    
    # Step 3: Penalty based on crowding of binding constraint
    # For each constraint c, compute the total prize of items that bind to c
    constraint_crowding = np.zeros(n_constraints)
    for c in range(n_constraints):
        mask = (binding_constraint == c)
        constraint_crowding[c] = np.sum(base_score[mask])
    
    # Normalize crowding to [0, 1] range for penalty
    max_crowding = np.max(constraint_crowding) if np.max(constraint_crowding) > 0 else 1.0
    normalized_crowding = constraint_crowding / max_crowding
    
    # For each item, get the crowding of its binding constraint
    item_crowding = normalized_crowding[binding_constraint]  # shape (n_items,)
    
    # Penalty: reduce score for items in crowded constraints
    # Use a multiplicative penalty: score * (1 - alpha * crowding)
    alpha = 0.5  # Penalty strength
    penalty = 1.0 - alpha * item_crowding
    penalty = np.clip(penalty, 0.1, 1.0)  # Ensure positive
    
    # Final score
    scores = base_score * penalty
    
    # Step 4: Power amplification to sharpen distribution
    # Normalize scores to [0, 1] first, then apply power
    max_score = np.max(scores) if np.max(scores) > 0 else 1.0
    min_score = np.min(scores)
    range_score = max_score - min_score
    if range_score > 1e-12:
        normalized_scores = (scores - min_score) / range_score
    else:
        normalized_scores = np.ones_like(scores)
    
    # Apply power to sharpen: higher power = more focus on top items
    power = 2.0
    amplified = np.power(normalized_scores, power)
    
    # Ensure all values are positive (as per task requirement: non-positive become 1e-9)
    amplified = np.where(amplified > 0, amplified, 1e-9)
    
    return amplified
# EVOLVE-BLOCK-END
