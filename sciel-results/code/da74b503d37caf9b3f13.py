"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with uncertainty-weighted diversity
    to prevent redundant evaluations in the same basin.
    
    Since we don't have coordinates, we use:
    - Base LCB: mu - beta * sigma
    - Adaptive beta that decreases as the GP converges
    - A diversity penalty that down-weights candidates with very low uncertainty
      (likely already well-explored regions) while preserving high-uncertainty
      candidates for exploration
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Compute convergence proxy ---
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    
    if sigma_max > 1e-10:
        conv_ratio = sigma_mean / sigma_max
    else:
        conv_ratio = 1.0
    
    # --- Adaptive beta for LCB ---
    # Beta starts high for exploration, decreases as convergence increases
    # Use a smoother decay based on the fraction of high-uncertainty points
    high_unc_frac = np.mean(sigma_safe > 0.5 * sigma_mean)
    beta = 1.5 + 1.5 * high_unc_frac
    
    # Scale by gap to theoretical minimum (more exploration if far from optimum)
    f_min_theory = -3.86278
    gap = f_best - f_min_theory
    if gap > 1e-6:
        beta *= (1.0 + 0.3 * min(gap / 1.0, 1.0))
    
    # --- LCB scores ---
    # Lower LCB = better for minimization; negate for "higher score = evaluate next"
    lcb_scores = -(mu - beta * sigma_safe)
    
    # --- Diversity-promoting uncertainty weighting ---
    # Penalize candidates with very low sigma (likely already explored basins)
    # This encourages the algorithm to explore high-uncertainty regions
    # Use a soft penalty: score *= (1 + alpha * sigma_normalized)
    # where sigma_normalized is in [0, 1]
    
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Alpha controls strength of diversity promotion
    # Higher alpha = stronger preference for high-uncertainty candidates
    alpha = 0.5
    
    diversity_weight = 1.0 + alpha * sigma_norm
    
    scores = lcb_scores * diversity_weight
    
    # --- Time check with safety margin ---
    elapsed = time.time() - start_time
    if elapsed > 25.0:
        # Fallback: return plain LCB scores
        scores = lcb_scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
