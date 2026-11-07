import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability: avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Use Upper Confidence Bound (UCB) acquisition function
    # UCB = mu + kappa * sigma
    # We want to minimize, so we want to find points with low mu and high uncertainty.
    # Standard UCB for minimization is: mu - kappa * sigma (if we want to maximize score for selection)
    # Or: score = -mu + kappa * sigma
    
    # Let's try a balance between exploitation (low mu) and exploration (high sigma).
    # A common choice for kappa is 1.0 or sqrt(2 * log(t)) where t is the number of iterations.
    # Since we don't have t, we use a fixed kappa.
    # Given the short horizon (30 evals), a moderate kappa is appropriate.
    # Let's try kappa = 1.0 first. If it's too exploratory, we can reduce it.
    # If it's too exploitative, we can increase it.
    
    # Actually, for minimization, the "improvement" over f_best is (f_best - mu).
    # UCB is often formulated as: mu + kappa * sigma (for maximization).
    # For minimization: mu - kappa * sigma.
    # We want to select the candidate with the highest score.
    # So score = -mu + kappa * sigma.
    
    kappa = 1.0
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
