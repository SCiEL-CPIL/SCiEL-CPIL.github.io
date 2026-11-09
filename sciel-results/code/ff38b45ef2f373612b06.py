"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Relative Lower Confidence Bound (RLCB).
    
    This approach normalizes the mean and standard deviation to [0, 1] 
    based on the current batch of candidates before applying the LCB formula.
    This makes the exploration/exploitation balance adaptive to the scale
    of the posterior predictions, which can change as the GP learns more
    about the landscape.
    
    Score = -mu_norm + beta * sigma_norm
    Higher scores indicate candidates with relatively lower expected values 
    or relatively high uncertainty.
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of RLCB scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    if time.time() > deadline:
        return np.zeros(n)
    
    # Sanitize inputs
    eps = 1e-8
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Structural Change: Relative Normalization
    # Normalize mu and sigma to [0, 1] range based on min/max in the current candidate set.
    # This adapts the effective beta to the current state of the GP.
    
    mu_min = np.min(mu_clean)
    mu_max = np.max(mu_clean)
    mu_range = mu_max - mu_min
    if mu_range < eps:
        mu_norm = np.zeros(n)
    else:
        mu_norm = (mu_clean - mu_min) / mu_range
        
    sigma_min = np.min(sigma_clean)
    sigma_max = np.max(sigma_clean)
    sigma_range = sigma_max - sigma_min
    if sigma_range < eps:
        sigma_norm = np.zeros(n)
    else:
        sigma_norm = (sigma_clean - sigma_min) / sigma_range
        
    # LCB Parameter
    # Since we are using normalized values, beta acts as a weight between 
    # relative exploitation and relative exploration.
    # A value of 1.0 gives equal weight to the relative scales.
    beta = 1.0
    
    # Compute RLCB score
    # We want to minimize mu, so -mu_norm.
    # We want to maximize sigma, so + beta * sigma_norm.
    scores = -mu_norm + beta * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
