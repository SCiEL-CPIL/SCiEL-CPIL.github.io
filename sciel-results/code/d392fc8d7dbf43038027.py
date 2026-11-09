import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid approach combining Expected Improvement (EI) with cost normalization
    and an adaptive exploration bonus to balance exploitation and exploration.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0)
    
    # Cost normalization: divide by cost to get "improvement per unit cost"
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    cost_aware_ei = ei / cost_safe
    
    # Adaptive exploration: scale exploration weight based on how much improvement potential exists
    # When f_best is high (early), we want more exploration; when low (late), less
    # Use the typical scale of the problem: Levy 2D min is 0, max is ~400
    # Normalize by a reference scale to make this adaptive
    ref_scale = max(abs(f_best), 1.0)  # at least 1.0 to avoid division issues
    improvement_potential = f_best / ref_scale  # 1.0 if f_best is the reference, less if we've improved
    
    # Exploration weight: starts high when f_best is large, decreases as we improve
    # Base weight of 0.15, scaled by improvement potential
    # Also add a small component based on relative uncertainty
    mean_sigma = np.mean(sigma_safe)
    rel_uncertainty = mean_sigma / ref_scale if ref_scale > 0 else 0.0
    
    # Adaptive exploration weight: higher when we're early or uncertainty is high
    exploration_weight = 0.15 * (0.5 + 0.5 * improvement_potential) + 0.05 * rel_uncertainty
    
    # Exploration bonus: uncertainty per unit cost, scaled by adaptive weight
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Final score: combine cost-aware EI with adaptive exploration bonus
    score = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
