import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score indicates a more promising point (lower expected value or high uncertainty)
    
    # Beta controls the exploration-exploitation trade-off
    # A higher beta favors exploration (uncertainty)
    # For a short budget (30 evals) and a multimodal function, strong exploration is beneficial
    beta = 2.5
    
    # Ensure numerical stability
    # sigma should be >= 0, but clamp to avoid any negative values from numerical noise
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
