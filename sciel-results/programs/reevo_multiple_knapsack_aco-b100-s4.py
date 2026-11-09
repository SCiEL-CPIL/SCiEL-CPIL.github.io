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
        total_prizes = prizes.sum(axis=0)
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        total_prizes = prizes
    else:
        total_prizes = prizes.reshape(-1)
        if total_prizes.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_prizes.shape[0])
            tmp[:n] = total_prizes[:n]
            total_prizes = tmp
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        total_weights = weights.sum(axis=0)
    elif weights.ndim == 1 and weights.shape[0] == 100:
        total_weights = weights
    else:
        total_weights = weights.reshape(-1)
        if total_weights.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_weights.shape[0])
            tmp[:n] = total_weights[:n]
            total_weights = tmp
    
    # Use ratio of total prize to total weight, raised to a power > 1 to sharpen
    # This focuses the ACO on items with high prize-to-weight ratio
    ratio = total_prizes / (total_weights + 1e-9)
    
    # Sharpen with power 2.5 to create stronger differentiation
    scores = ratio ** 2.5
    
    # Ensure all scores are positive (task says non-positive become 1e-9)
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
