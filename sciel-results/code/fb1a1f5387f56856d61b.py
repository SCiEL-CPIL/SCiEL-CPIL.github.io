"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ACO-based bin packing.
    
    The heuristic favors pairs (i, j) where placing item j after item i
    leaves little wasted space in the bin, i.e., where the combined size
    of the two items is close to the bin capacity (but not exceeding it).
    
    We use a sharp decay function that strongly rewards near-perfect fits.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pairwise sum: pair[i, j] = d[i] + d[j]
    pair = d[:, None] + d[None, :]
    
    # Remaining space after placing i and j together
    # We want this to be small (close to 0) but >= 0
    remaining = capacity - pair
    
    # Only valid if remaining >= 0 (i.e., pair <= capacity)
    valid = remaining >= 0
    
    # For valid pairs, compute attractiveness based on how well they fit.
    # A pair that fills the bin perfectly (remaining=0) gets the highest score.
    # We use an exponential decay: exp(-remaining / scale)
    # Sharp decay (small scale) strongly favors near-perfect fits.
    scale = 3.0
    power = 3.0
    
    # For valid pairs: attractiveness = (remaining + epsilon)^(-power) * exp(-remaining / scale)
    # Actually, let's think more carefully. We want:
    # - Pairs that fit (remaining >= 0) to have high attractiveness
    # - Pairs that nearly fill the bin (remaining close to 0) to have the highest
    # - Pairs that leave lots of space to have lower attractiveness
    # - Pairs that don't fit to have minimal attractiveness (1e-9)
    
    # Use: for valid pairs, h = 1.0 / (1.0 + remaining/scale)^power
    # This gives h=1 when remaining=0, and decays as remaining increases.
    # With scale=3 and power=3, this is quite sharp.
    
    # Alternative: h = exp(-remaining / scale) for valid pairs
    # This gives h=1 when remaining=0, and decays exponentially.
    
    # Let's combine: h = (1.0 + remaining/scale)^(-power)
    # When remaining=0: h=1
    # When remaining=3: h=0.125
    # When remaining=10: h=0.0046
    
    safe_remaining = np.where(valid, remaining, 0.0)
    base = 1.0 + safe_remaining / scale
    h_valid = base ** (-power)
    
    # For invalid pairs, use 1e-9
    h = np.where(valid, h_valid, 1e-9)
    
    # Ensure no non-finite values
    h = np.where(np.isfinite(h), h, 1e-9)
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
