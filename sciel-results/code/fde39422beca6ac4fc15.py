"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level state for a simple, robust GP-based acquisition
_T0 = time.time()
_DEADLINE = 315.0  # safety margin before 330s kill

# We will use a lightweight GP with Matern-5/2, length-scale 0.2 on [0,1]^2,
# noise 1e-6, standardized targets. However, since the acquisition function
# only receives mu, sigma, f_best, cost (no coordinates or data), we cannot
# recompute the GP inside acquisition. The evaluator supplies mu and sigma
# computed by its own GP. So our job is purely: given (mu, sigma, f_best, cost),
# return 2000 scores.

# The key insight: the evaluator's loop is:
#   1. Start with 5 uniform random points (cost counted).
#   2. While total cost < 30:
#        a. Score 2000 new uniform candidates with acquisition(mu, sigma, f_best, cost)
#        b. Evaluate the best one.
# So acquisition is called repeatedly with the same 2000 candidate points
# but updated mu, sigma, f_best, cost.

# Since we don't have coordinates, we must design an acquisition function
# that works well in aggregate. The cost is 1 + mean(x scaled to [0,1]),
# so points near the center (0.5, 0.5) have cost ~1.5, corners have cost ~2.

# Let's try a different structural approach: use a hybrid of Expected Improvement
# and Probability of Improvement, weighted by cost efficiency, with adaptive
# exploration based on the spread of sigma.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std dev predictions
    - f_best: scalar, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of scores (higher = more desirable to evaluate)
    """
    # Check time budget
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE:
        # Return a safe default: prioritize low cost
        return 1.0 / cost
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement with adaptive xi
    # Use a small xi that adapts to the scale of f_best
    xi = 0.005 * max(abs(f_best), 1e-6)
    
    improvement = f_best - mu - xi
    z = improvement / sigma
    
    # Expected improvement
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Blend EI and PI with a weight that depends on how uncertain we are
    # When sigma is large relative to |f_best|, explore more (weight PI higher)
    # When sigma is small, exploit more (weight EI higher)
    sigma_scale = np.median(sigma) + 1e-9
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = sigma_scale / f_scale
    
    # Blend weight: more exploration when uncertain
    w_explore = np.clip(0.5 + 0.5 * np.log1p(uncertainty_ratio), 0.3, 0.7)
    
    score = w_explore * pi + (1 - w_explore) * ei
    
    # Normalize scores to [0, 1] range for stability
    score_min = np.min(score)
    score_max = np.max(score)
    if score_max - score_min > 1e-12:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.ones_like(score) * 0.5
    
    # Cost-aware weighting: divide by cost, but use a soft penalty
    # to avoid over-penalizing slightly higher cost points
    cost_penalty = np.power(cost, 0.8)  # sublinear penalty
    
    final_score = score / cost_penalty
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
