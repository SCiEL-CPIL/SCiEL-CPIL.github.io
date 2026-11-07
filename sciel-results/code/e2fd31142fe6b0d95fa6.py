"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we return before 330s
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Probability of Improvement
    (PI) and Expected Improvement (EI), normalized by evaluation cost.
    
    The score is (w * PI + (1-w) * EI) / cost.
    PI focuses on high-confidence gains, while EI captures the magnitude of improvement.
    A fixed epsilon threshold is used for PI to balance exploration and exploitation.
    """
    # Check deadline - if close to limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 320.0:  # Safety margin before 330s kill
        # Return simple score based on mu (lower is better) normalized by cost
        cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Fixed epsilon for the improvement threshold
    # Based on the finding that epsilon=0.01 works well for cost-aware optimization
    epsilon = 0.01
    target = f_best - epsilon
    
    # Calculate z-score for the improvement threshold
    z = (target - mu) / sigma_safe
    
    # Clip z to prevent numerical issues
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < target) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # Expected Improvement:
    # EI = (target - mu) * Phi(z) + sigma * phi(z)
    pdf_z = norm.pdf(z)
    ei = (target - mu) * prob_improve + sigma_safe * pdf_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize PI and EI to comparable scales
    # PI is already in [0, 1]
    # EI is in objective units, so we scale it to be comparable to PI
    # We can use the median EI as a scale factor
    median_ei = np.median(ei)
    if median_ei > 1e-8:
        ei_normalized = ei / median_ei
    else:
        ei_normalized = ei
    
    # Blend PI and EI
    # Weight of 0.7 for PI and 0.3 for EI
    # This emphasizes high-confidence improvements while still considering magnitude
    w_pi = 0.7
    w_ei = 0.3
    combined_score = w_pi * prob_improve + w_ei * ei_normalized
    
    # Normalize by cost to get cost-aware score
    scores = combined_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
