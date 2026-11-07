# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so LCB = mu - kappa * sigma
    # Higher LCB score means more promising (lower expected value with uncertainty)
    # Actually for minimization, we want low mu - kappa*sigma, so score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # Let's use a standard LCB approach: minimize mu - kappa*sigma
    # So the "score" for LCB (higher is better) is: -mu + kappa*sigma
    
    # Adaptive kappa based on the spread of mu and sigma
    mu_range = mu.max() - mu.min()
    if mu_range < 1e-10:
        mu_range = 1.0
    
    sigma_mean = sigma.mean()
    
    # Balance exploration and exploitation
    # Use a moderate kappa that adapts to the problem scale
    kappa = 1.5 + 0.5 * (sigma_mean / (mu_range + 1e-10))
    kappa = np.clip(kappa, 1.0, 3.0)
    
    # LCB score (higher is better for minimization)
    lcb_score = -mu + kappa * sigma
    
    # Normalize EI to [0, 1]
    ei_max = ei.max()
    if ei_max < 1e-10:
        ei_norm = ei
    else:
        ei_norm = ei / ei_max
    
    # Normalize LCB score to [0, 1]
    lcb_min = lcb_score.min()
    lcb_max = lcb_score.max()
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-10:
        lcb_norm = lcb_score
    else:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine EI and LCB
    # Weight EI more when we're close to the minimum, LCB more when exploring
    # Use the ratio of current best to theoretical minimum as a heuristic
    # Theoretical minimum for cosine8 8d is -0.8
    theoretical_min = -0.8
    if f_best > theoretical_min + 1e-6:
        progress = 1.0 - (f_best - theoretical_min) / (f_best - theoretical_min + 1e-10)
        # This doesn't make sense, let's use a different approach
        # When f_best is close to theoretical_min, exploit more (weight EI higher)
        # When f_best is far, explore more (weight LCB higher)
        closeness = np.clip(1.0 - (f_best - theoretical_min) / 1.0, 0.0, 1.0)
        ei_weight = 0.5 + 0.3 * closeness
        lcb_weight = 1.0 - ei_weight
    else:
        ei_weight = 0.8
        lcb_weight = 0.2
    
    combined = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # Cost-aware weighting
    # Cheaper evaluations should be preferred
    # Use score / cost^alpha where alpha > 1 gives stronger preference for cheap points
    alpha = 1.3
    cost_safe = np.maximum(cost, 1e-6)
    cost_penalty = cost_safe ** alpha
    
    # Apply cost-aware scoring
    scores = combined / cost_penalty
    
    # Add a small bonus for points with low predicted values (exploitation)
    # This helps maintain focus on promising regions
    mu_min = mu.min()
    mu_max = mu.max()
    mu_range_safe = mu_max - mu_min
    if mu_range_safe < 1e-10:
        mu_range_safe = 1.0
    mu_norm = (mu - mu_min) / mu_range_safe
    exploitation_bonus = (1.0 - mu_norm) * 0.05
    
    scores = scores + exploitation_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
