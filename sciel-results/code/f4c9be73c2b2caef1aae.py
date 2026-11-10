"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure we have the right shape
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        # Sum prizes across constraints to get total prize per item
        total_prizes = prizes.sum(axis=0)
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        total_prizes = prizes
    else:
        # Fallback: try to reshape or handle
        total_prizes = prizes.reshape(-1)
        if total_prizes.shape[0] != 100:
            # Pad or truncate to 100
            tmp = np.zeros(100)
            n = min(100, total_prizes.shape[0])
            tmp[:n] = total_prizes[:n]
            total_prizes = tmp
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        # Compute bottleneck slack: min across constraints of (capacity - weight) / weight
        # Capacity is 25 for each constraint
        # This measures how much room an item leaves in its most constraining dimension
        capacity = 25.0
        # For each constraint c and item i: slack_c_i = (capacity - weights[c, i]) / (weights[c, i] + 1e-9)
        # Then bottleneck_slack_i = min_c slack_c_i
        slack = (capacity - weights) / (weights + 1e-9)
        bottleneck_slack = slack.min(axis=0)
        
        # Also compute total weights for density ratio
        total_weights = weights.sum(axis=0)
    elif weights.ndim == 1 and weights.shape[0] == 100:
        total_weights = weights
        bottleneck_slack = np.ones(100)  # No bottleneck info for 1D
    else:
        total_weights = weights.reshape(-1)
        if total_weights.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_weights.shape[0])
            tmp[:n] = total_weights[:n]
            total_weights = tmp
        bottleneck_slack = np.ones(100)
    
    # Compute density ratio
    density = total_prizes / (total_weights + 1e-9)
    
    # Combine density with bottleneck slack
    # The bottleneck slack directly penalizes items that consume disproportionate capacity
    # in their most constraining dimension
    # Use a smooth combination: multiply density by (1 + bottleneck_slack)
    # This rewards items that both have good density AND fit easily in tight constraints
    scores = density * (1.0 + bottleneck_slack)
    
    # Sharpen with power 1.5 to create stronger differentiation while preserving structure
    scores = scores ** 1.5
    
    # Ensure all scores are positive (task says non-positive become 1e-9)
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
