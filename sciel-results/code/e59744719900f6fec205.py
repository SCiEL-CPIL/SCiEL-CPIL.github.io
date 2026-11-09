"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    The blend weight is adapted based on the median uncertainty (sigma) of the GP,
    favoring exploration (UCB) when uncertainty is high and exploitation (EI)
    when uncertainty is low.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lowest)
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Safety: ensure finite inputs and correct types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    # z-score for EI
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in CDF/PDF
    z_clipped = np.clip(z, -40, 40)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1] for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # Optimistic value: mu - kappa * sigma.
    # We want to maximize the acquisition score, so we negate the optimistic value:
    # Score_UCB = -(mu - kappa * sigma) = -mu + kappa * sigma
    kappa = 2.5  # Exploration constant
    
    ucb = -mu + kappa * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
        
    # 3. Dynamic Blending Weight
    # Determine how much to trust UCB (exploration) vs EI (exploitation).
    # High median sigma -> High Uncertainty -> Explore more (higher weight on UCB).
    # Low median sigma -> Low Uncertainty -> Exploit more (lower weight on UCB).
    
    sigma_median = np.median(sigma_safe)
    
    # Map sigma_median to a weight w in [0.2, 0.8]
    # If sigma_median is 0.1, w ~ 0.2 (mostly EI)
    # If sigma_median is 0.5, w ~ 0.8 (mostly UCB)
    # If sigma_median > 0.5, w = 0.8
    # If sigma_median < 0.1, w = 0.2
    
    # Linear mapping from [0.1, 0.5] to [0.2, 0.8]
    # w = 0.2 + (sigma_median - 0.1) * (0.8 - 0.2) / (0.5 - 0.1)
    # w = 0.2 + (sigma_median - 0.1) * 1.5
    w = 0.2 + (sigma_median - 0.1) * 1.5
    w = np.clip(w, 0.2, 0.8)
    
    # 4. Combine
    blend = w * ucb_norm + (1 - w) * ei_norm
    
    # 5. Cost-Aware Score
    scores = blend / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 290:
        pass # Should not happen for N=2000
        
    return scores
# EVOLVE-BLOCK-END
