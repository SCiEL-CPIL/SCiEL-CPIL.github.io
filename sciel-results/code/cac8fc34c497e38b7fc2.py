"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Hard cost-gated Expected Improvement:
    # Instead of smooth EI/cost, use a threshold gate that heavily damps
    # expensive candidates, forcing the search to exhaust uncertainty in
    # cheap regions before expanding to expensive ones.
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Hard cost gate: use a percentile-based threshold
    # Candidates in the cheapest 60% get full EI; others are damped
    # This forces search to concentrate in cheap regions first
    cost_threshold = np.percentile(cost, 60)
    
    # Smooth gate: full score below threshold, damped above
    # Use a soft step to avoid discontinuities
    gate_width = 0.15  # width of the transition region
    gate = 1.0 / (1.0 + np.exp(10.0 * (cost - cost_threshold) / (gate_width + 1e-9)))
    
    # Apply gate: candidates above threshold get heavily reduced score
    # The gate value is in [0, 1], with ~1 below threshold and ~0 above
    scores = ei * gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
