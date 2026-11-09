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
    capacity = 25.0
    
    # Compute aggregate metrics
    total_prize = p.sum(axis=0)  # (100,)
    total_weight = w.sum(axis=0)  # (100,)
    max_weight_per_item = w.max(axis=0)  # (100,)
    min_weight_per_item = w.min(axis=0)  # (100,)
    
    # Prize-to-total-weight ratio
    ratio = total_prize / (total_weight + 1e-9)
    
    # Penalty for items that are heavy in any single constraint
    # An item with high weight in one constraint limits what else can be added
    bottleneck_penalty = max_weight_per_item  # higher = more restrictive
    
    # Combined score: high ratio, low bottleneck
    # Use a multiplicative form to balance
    base_score = ratio * (1.0 / (1.0 + bottleneck_penalty))
    
    # Additionally, consider the "efficiency" per constraint
    # For each constraint, compute prize/weight for that constraint
    # Then take the minimum (weakest constraint) as a robustness measure
    if n_constraints > 1:
        constraint_ratios = p / (w + 1e-9)  # (n_constraints, 100)
        min_constraint_ratio = constraint_ratios.min(axis=0)  # (100,)
        # Blend with the total ratio
        base_score = 0.5 * base_score + 0.5 * (min_constraint_ratio / (1.0 + bottleneck_penalty))
    else:
        base_score = base_score
    
    # Sharpen the distribution to create stronger differentiation
    # Normalize to [0, 1] first
    bmin = base_score.min()
    bmax = base_score.max()
    if bmax > bmin:
        normalized = (base_score - bmin) / (bmax - bmin)
    else:
        normalized = np.ones(n_items) / n_items
    
    # Apply power to sharpen - higher power makes top items much more attractive
    power = 3.0
    scores = normalized ** power
    
    # Add a small floor to avoid zeros
    scores = scores + 1e-6
    
    # Ensure all scores are positive
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
