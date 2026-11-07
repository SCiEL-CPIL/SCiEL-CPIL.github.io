import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound
    # Score = (-mu + kappa * sigma) / cost
    # We want to maximize this.
    # f_best is the current best (lowest) value.
    # mu is the mean prediction.
    # sigma is the std prediction.
    
    # Scale the objective term to be comparable to cost?
    # f_best is in objective units. cost is in [1, 2].
    # If f_best is large, -mu is large negative.
    # We want to minimize f. So -mu is "good" if mu is low.
    # sigma is "good" if it's high (uncertainty).
    
    # Let's normalize by f_best to make it scale-invariant?
    # If f_best is 0, we can't divide.
    # f_best is the lowest observed. Powell min is 0.
    # If f_best is close to 0, the search is done.
    
    # Let's use a fixed kappa.
    kappa = 2.5
    
    # Numerator: -mu + kappa * sigma
    # We want to maximize this.
    # If mu is low (good), -mu is high (positive).
    # If sigma is high (uncertain), kappa*sigma is high.
    
    num = -mu + kappa * sigma
    
    # Denominator: cost
    # Cost is in [1, 2].
    # We divide by cost to penalize expensive points.
    
    score = num / (cost + 1e-9)
    
    return score
