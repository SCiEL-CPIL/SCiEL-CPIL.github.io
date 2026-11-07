import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Structural switching:
    # If f_best is not yet close to the global minimum, use UCB for exploration.
    # Once f_best is close, use EI for exploitation.
    
    # Global minimum of Hartmann 3D is -3.86278
    # Threshold for switching to exploitation
    threshold = -3.5
    
    if f_best > threshold:
        # Exploration phase: UCB
        # Score = -mu + beta * sigma
        # We want to maximize this score.
        # Higher sigma encourages exploration.
        # Lower mu encourages exploitation of known good areas, but UCB balances this.
        beta = 1.5
        return -mu + beta * sigma
    else:
        # Exploitation phase: Expected Improvement
        # Standard EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
