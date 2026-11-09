# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    score = (kappa * sigma - mu) / cost
    
    where kappa is dynamically tuned based on the ratio of remaining budget
    to current cost to shift the exploration-exploitation balance.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate remaining budget using the mean cost of candidates as a proxy
    # for typical evaluation cost, and assume we've used roughly half the budget
    # as a heuristic for mid-search behavior.
    mean_cost = np.mean(cost)
    # Assume total budget is 30, and estimate spent as mean_cost * some factor
    # We use a simple heuristic: remaining_fraction = 1 - (mean_cost / 30) * 2
    # This gives ~1 at start, ~0 at end
    remaining_fraction = np.clip(1.0 - (mean_cost / 30.0) * 2.0, 0.0, 1.0)
    
    # Dynamic kappa: higher when more budget remains (explore more),
    # lower when budget is nearly exhausted (exploit more)
    kappa_min = 0.5
    kappa_max = 4.0
    
    # Use a smooth transition based on remaining budget fraction
    kappa = kappa_min + (kappa_max - kappa_min) * remaining_fraction
    
    # For minimization, we want to select points with low mu and high sigma.
    # The UCB score for minimization (higher is better) is:
    # score = (kappa * sigma - mu) / cost
    #
    # This favors points with high uncertainty (sigma) and low predicted mean (mu),
    # normalized by the cost of evaluation.
    
    score = (kappa * sigma - mu) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
