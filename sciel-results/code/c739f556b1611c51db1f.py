"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Item goes to highest-scoring bin. Lower total excess is better.
    
    Strategy: Rank-based scoring with smooth penalty for small remainders.
    - Compute remainder after placing item in each feasible bin
    - Score by rank (fewer bins have smaller remainder = better)
    - Apply smooth penalty for remainders that are too small to be useful
    """
    remainders = bins - item
    feasible = remainders >= 0
    
    # For infeasible bins, assign very low score
    scores = np.full(len(bins), -1e10)
    
    if not np.any(feasible):
        return scores
    
    feas_remainders = remainders[feasible]
    
    # Rank-based score: count how many feasible bins have smaller remainder
    # Higher rank = fewer bins with smaller remainder = "tighter" fit relative to others
    # But we want to AVOID the tightest fits if they create unusable gaps
    
    # Compute rank (0 = smallest remainder, n-1 = largest remainder)
    # Use argsort for stable ranking
    order = np.argsort(feas_remainders)
    ranks = np.empty(len(feas_remainders), dtype=np.float64)
    ranks[order] = np.arange(len(feas_remainders))
    
    # Normalize rank to [0, 1]
    n_feas = len(feas_remainders)
    if n_feas > 1:
        norm_ranks = ranks / (n_feas - 1)
    else:
        norm_ranks = np.zeros_like(ranks)
    
    # Penalty for small remainders: if remainder is very small relative to item,
    # it's likely wasted. Use a smooth sigmoid-like penalty.
    # Scale: remainders < 0.1 * item get penalized
    penalty = 1.0 / (1.0 + np.exp(20.0 * (feas_remainders / max(item, 1e-9) - 0.15)))
    
    # Combine: prefer higher rank (more space left) but penalize very small remainders
    # Actually, let's reconsider: we want to AVOID creating tiny gaps
    # So we should PENALIZE low remainders (tight fits)
    # Score = norm_rank - penalty_weight * penalty
    # This makes bins with moderate remainders most attractive
    
    penalty_weight = 0.5
    combined = norm_ranks - penalty_weight * penalty
    
    # Assign scores back
    scores[feasible] = combined
    
    return scores
# EVOLVE-BLOCK-END
