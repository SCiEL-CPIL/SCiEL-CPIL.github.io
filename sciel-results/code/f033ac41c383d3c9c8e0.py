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
        max_weights = weights.max(axis=0)
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        total_prizes = prizes
        max_weights = np.ones(100)
    else:
        total_prizes = prizes.reshape(-1)
        if total_prizes.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_prizes.shape[0])
            tmp[:n] = total_prizes[:n]
            total_prizes = tmp
        max_weights = np.ones(100)
    
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
    
    # Compute prize-to-weight ratio
    ratio = total_prizes / (total_weights + 1e-9)
    
    # Normalize by geometric mean to make scale-invariant
    log_ratio = np.log(ratio + 1e-9)
    geo_mean = np.exp(np.mean(log_ratio))
    ratio_norm = ratio / (geo_mean + 1e-9)
    
    # Apply power to sharpen (moderate sharpness)
    score = ratio_norm ** 2.0
    
    # Add penalty for items that are heavy in their worst constraint
    # This encourages selection of "flexible" items that don't bottleneck
    penalty = (1.0 / (1.0 + max_weights)) ** 0.3
    score = score * penalty
    
    # Ensure all scores are positive
    score = np.where(score > 0, score, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
