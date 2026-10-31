"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import math

def priority(vector, n):
    """
    Static priority for cap set greedy construction.
    
    Heuristic: prefer vectors with:
    1. Coordinate sum near n (balanced)
    2. High coordinate spread (variance) - more "generic" vectors
    3. Deterministic hash tiebreaker for stability
    
    This guides greedy insertion toward balanced, spread-out vectors
    that are less likely to form affine lines with other selected vectors.
    """
    # Coordinate sum: prefer vectors centered around n
    s = sum(vector)
    sum_dev = abs(s - n)
    
    # Coordinate variance/spread: prefer vectors with mixed coordinates
    # Mean is s/n, variance = E[x^2] - (E[x])^2
    sum_sq = sum(x * x for x in vector)
    mean = s / n
    variance = sum_sq / n - mean * mean
    
    # Score: minimize sum_dev, maximize variance
    # Use a smooth combination
    sum_penalty = sum_dev / n  # normalize to [0, 1]
    spread_bonus = variance  # in [0, 1] roughly
    
    # Primary: spread (higher is better), Secondary: sum balance (lower dev is better)
    # Combine: spread_bonus - 0.5 * sum_penalty
    score = spread_bonus - 0.5 * sum_penalty
    
    # Deterministic tiebreaker using hash of the vector
    # This ensures consistent ordering and can break near-ties in a structured way
    vec_bytes = bytes(vector)
    h = int(hashlib.md5(vec_bytes).hexdigest()[:8], 16)
    # Scale hash to a small perturbation to not dominate the main score
    hash_perturb = (h % 10000) / 10000.0 * 0.01
    
    return score + hash_perturb
# EVOLVE-BLOCK-END
