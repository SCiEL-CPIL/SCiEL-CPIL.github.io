import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use UCB (Upper Confidence Bound) for minimization.
    # The score represents the "potential" of the point to be the new minimum.
    # For minimization, we look at the lower bound of the prediction: mu - kappa * sigma.
    # We want to maximize the negative of this lower bound: kappa * sigma - mu.
    
    # Cost adjustment:
    # The cost is between 1 and 2.
    # Dividing by cost penalizes expensive points.
    # Since the global minimum of Styblinski-Tang is in the 'expensive' region (high coords),
    # a strong penalty might hinder convergence.
    # However, standard cost-aware BO divides by cost.
    # Let's stick to / cost but use UCB which is more robust.
    
    kappa = 2.5
    scores = (kappa * sigma - mu) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
