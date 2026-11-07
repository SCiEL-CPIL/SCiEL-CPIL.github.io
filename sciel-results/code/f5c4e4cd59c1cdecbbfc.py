"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Per-constraint efficiency
    eff = prizes / (weights + 1e-9)  # (5, 100)
    
    # Geometric mean of per-constraint efficiencies:
    # smooth surrogate that rewards items efficient across all constraints,
    # penalizing items that are imbalanced (good in one constraint, poor in another)
    log_eff = np.log(eff + 1e-9)
    geo_mean = np.exp(log_eff.mean(axis=0))  # (100,)
    
    # Scale by total prize to maintain comparable magnitude to baseline
    total_prize = prizes.sum(axis=0)
    
    heuristic = geo_mean * total_prize
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
