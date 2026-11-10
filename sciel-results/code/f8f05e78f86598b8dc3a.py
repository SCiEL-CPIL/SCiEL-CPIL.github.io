"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape robustly
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        p = prizes
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        p = prizes.reshape(1, 100)
    else:
        flat = prizes.reshape(-1)
        if flat.shape[0] >= 100:
            p = flat[:100].reshape(1, 100)
        else:
            p = np.zeros((1, 100))
            p[0, :flat.shape[0]] = flat
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        w = weights
    elif weights.ndim == 1 and weights.shape[0] == 100:
        w = weights.reshape(1, 100)
    else:
        flat = weights.reshape(-1)
        if flat.shape[0] >= 100:
            w = flat[:100].reshape(1, 100)
        else:
            w = np.zeros((1, 100))
            w[0, :flat.shape[0]] = flat
    
    n_items = 100
    n_constraints = p.shape[0]
    
    # Compute aggregate metrics
    total_prize = p.sum(axis=0)  # (100,)
    total_weight = w.sum(axis=0)  # (100,)
    max_weight = w.max(axis=0)  # (100,)
    
    # Composite weight: captures both the worst-case constraint (max) 
    # and total resource consumption (sum)
    composite_weight = max_weight + 0.5 * total_weight
    
    # Prize-to-composite-weight ratio
    ratio = total_prize / (composite_weight + 1e-9)
    
    # Bottleneck slack: for each item, compute min_c (25 - w_ic) / (w_ic + 1e-9)
    # This measures how much "slack" remains in the tightest constraint if this item is added
    capacity = 25.0
    slack_per_constraint = (capacity - w) / (w + 1e-9)  # (n_constraints, 100)
    bottleneck_slack = slack_per_constraint.min(axis=0)  # (100,)
    
    # Ensure slack is non-negative (items that don't fit get very low slack)
    bottleneck_slack = np.maximum(bottleneck_slack, 0.0)
    
    # Combine ratio and bottleneck slack
    # The slack term rewards items that leave more room in their tightest constraint
    # Multiply ratio by (1 + slack) to boost items with high feasibility margin
    scores = ratio * (1.0 + bottleneck_slack)
    
    # Sharpen with power 2.5 to create differentiation while maintaining
    # a smooth distribution for the ACO
    scores = scores ** 2.5
    
    # Ensure all scores are positive (task says non-positive become 1e-9)
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
