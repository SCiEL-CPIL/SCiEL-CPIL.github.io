"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Add small epsilon to weights and prizes to avoid division by zero
    # We add to weights in denominator of efficiency, so we add to weights.
    # We add to prizes in numerator, so we add to prizes.
    eps = 1e-9
    
    # Calculate efficiency ratios: prize / weight
    # Shape: (5, 100)
    efficiencies = prizes / (weights + eps)
    
    # We want the harmonic mean of the efficiencies.
    # H = n / sum(1/x)
    # 1/x = weight / prize
    # Sum(1/x) = sum(weights / prizes)
    # If any prize is 0, 1/x is inf, H is 0.
    
    # Calculate inverse efficiencies (cost per prize unit)
    # If prize is 0, this becomes inf.
    inv_effs = (weights + eps) / (prizes + eps)
    
    # Sum across constraints (axis 0)
    sum_inv_effs = np.sum(inv_effs, axis=0)
    
    # Harmonic mean = 5 / sum_inv_effs
    heur = 5.0 / sum_inv_effs
    
    # Ensure values are finite and positive
    # If sum_inv_effs is 0 (impossible with eps), heur is inf.
    # If sum_inv_effs is inf, heur is 0.
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
