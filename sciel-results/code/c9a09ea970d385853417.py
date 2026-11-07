import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware, diversity-promoting acquisition function.
    # Uses a UCB-style score that explicitly weights predictive variance (sigma)
    # to encourage exploration of regions far from already-evaluated points,
    # as high sigma indicates high uncertainty/distance from data.
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization:
    # We want to maximize the score.
    # Term 1: (f_best - mu) favors points predicted to be better than current best.
    # Term 2: kappa * sigma favors points with high uncertainty (exploration/diversity).
    # This combination avoids the "zero EI" plateau and promotes domain coverage.
    kappa = 2.5
    
    # Base score: Exploitation + Exploration
    base_score = (f_best - mu) + kappa * sigma_safe
    
    # Normalize by cost to account for evaluation budget
    # Lower cost evaluations are preferred for similar expected gain
    scores = base_score / cost
    
    # Ensure scores are finite (handle any numerical issues)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
