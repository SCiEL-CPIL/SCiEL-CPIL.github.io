import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Robustness against numerical issues
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Aware UCB for minimization
    # We want to maximize the score.
    # Lower mu is better. Higher sigma is better (exploration).
    # Divide by cost to penalize expensive evaluations.
    kappa = 1.5
    scores = (-mu + kappa * sigma) / cost
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
