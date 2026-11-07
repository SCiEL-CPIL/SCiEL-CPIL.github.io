import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate statistics for normalization
    # We want to balance exploitation (low mu), exploration (high sigma), and cost (low cost)
    
    # Normalize mu: lower is better. We will use -mu.
    # Normalize sigma: higher is better.
    # Normalize cost: lower is better.
    
    # To avoid division by zero or large numbers, we use robust normalization (median/IQR or mean/std)
    # Using mean/std is standard.
    
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-8
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-8
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-8
    
    # Z-scores
    z_mu = (mu - mu_mean) / mu_std
    z_sigma = (sigma - sigma_mean) / sigma_std
    z_cost = (cost - cost_mean) / cost_std
    
    # We want to minimize mu, maximize sigma, minimize cost.
    # Score = -w_mu * z_mu + w_sigma * z_sigma - w_cost * z_cost
    
    # Weights:
    # w_mu: Exploitation weight.
    # w_sigma: Exploration weight (beta).
    # w_cost: Cost aversion weight.
    
    # For Griewank, exploration is important early on.
    # Let's start with equal weights, but maybe emphasize exploration slightly more.
    w_mu = 1.0
    w_sigma = 1.5  # Higher exploration weight
    w_cost = 0.5   # Cost penalty is moderate
    
    scores = -w_mu * z_mu + w_sigma * z_sigma - w_cost * z_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
