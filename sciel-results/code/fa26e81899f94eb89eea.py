# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    score = (f_best - mu + kappa * sigma) / cost
    
    where kappa is dynamically tuned based on the remaining budget to shift
    the exploration-exploitation balance as the search progresses.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate remaining budget. The total budget is 30.
    # We can estimate the number of evaluations done by looking at the cost array.
    # Each evaluation costs 1 + mean(x) where x is in [0,1]^3, so cost is in [1, 4].
    # The initial 5 points are uniform random, so their mean cost is about 1 + 1.5 = 2.5.
    # But we don't know exactly how many evaluations have been done.
    # 
    # A better proxy: use the spread of sigma to estimate how much information we have.
    # Early on, sigma is large everywhere. Later, sigma is small near evaluated points.
    # 
    # Alternative: Use the ratio of the current best predicted value to f_best.
    # 
    # Let's use a simpler heuristic: the mean sigma gives a sense of remaining uncertainty.
    # High mean sigma -> early stage -> more exploration (higher kappa)
    # Low mean sigma -> late stage -> less exploration (lower kappa)
    
    mean_sigma = np.mean(sigma)
    
    # Hartmann 3d range is roughly [-3.86, 0], so total range ~3.86
    # Normalize mean_sigma by this range to get a 0-1 scale
    # Early: mean_sigma ~ 1-2, late: mean_sigma ~ 0.1-0.5
    
    # Map mean_sigma to a progress indicator in [0, 1]
    # progress = 0 means early (high uncertainty), 1 means late (low uncertainty)
    sigma_scale = 2.0  # Typical early sigma
    progress = 1.0 - np.clip(mean_sigma / sigma_scale, 0, 1)
    
    # Dynamic kappa: high exploration early, low exploration late
    # kappa = kappa_min + (kappa_max - kappa_min) * (1 - progress)
    kappa_max = 2.5  # High exploration early
    kappa_min = 0.5  # Low exploration late
    
    kappa = kappa_min + (kappa_max - kappa_min) * (1.0 - progress)
    
    # For minimization, we want to pick points with low mu (good predicted value)
    # and high sigma (uncertainty). The acquisition score (higher is better) is:
    # score = (f_best - mu + kappa * sigma) / cost
    # 
    # f_best - mu: positive if mu is better than f_best (improvement potential)
    # kappa * sigma: uncertainty bonus
    # Dividing by cost: cost-awareness
    
    score = (f_best - mu + kappa * sigma) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
