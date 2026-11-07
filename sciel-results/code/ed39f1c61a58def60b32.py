import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound
    # We want to minimize f(x).
    # UCB lower bound: mu - kappa * sigma
    # We want to maximize the potential improvement: f_best - (mu - kappa * sigma)
    # Score = (f_best - mu + kappa * sigma) / cost
    kappa = 2.0
    score = (f_best - mu + kappa * sigma) / cost
    # Ensure finite
    return np.where(np.isfinite(score), score, 0.0)
