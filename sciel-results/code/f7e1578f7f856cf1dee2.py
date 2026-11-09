# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Strategy:
    - Compute standard Expected Improvement (EI)
    - Add an adaptive exploration term that decays as f_best improves
    - The exploration term uses a UCB-like structure with adaptive beta
    
    mu, sigma: shape (2000,), posterior mean and std
    f_best: lowest observed value
    Returns: 2000 scores, higher = more promising
    """
    # Safety: ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)
    
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: add a UCB-like term
    # beta decays as f_best approaches 0 (optimum)
    # Early on (f_best ~ 100-200), we want more exploration
    # Late (f_best ~ 0-5), we want exploitation
    
    # Beta schedule: starts high, decays smoothly
    # Using an exponential decay based on f_best
    # When f_best is large (early), beta is larger -> more exploration
    # When f_best is small (late), beta is smaller -> more exploitation
    
    # Scale parameter controls how fast exploration decays
    scale = 5.0
    beta = 1.0 * np.exp(-f_best / scale)
    
    # Also consider: if f_best is very small, pure EI is best
    # If f_best is large, add more exploration
    # The UCB term for minimization: -mu + beta * sigma
    # But we want to MAXIMIZE the acquisition, so:
    # score = EI + beta * sigma (the sigma term encourages exploration)
    
    # Alternative: use a weighted combination
    # Let's try: score = (1 - alpha) * EI + alpha * (beta * sigma)
    # where alpha depends on how uncertain we are overall
    
    # Mean sigma gives a sense of overall uncertainty
    mean_sigma = np.mean(sigma_safe)
    
    # If overall uncertainty is high, explore more
    # If low, exploit more
    # Normalize by f_best to get a relative measure
    if f_best > 1e-6:
        uncertainty_ratio = mean_sigma / f_best
    else:
        uncertainty_ratio = 1.0
    
    # Alpha: weight for exploration component
    # When uncertainty_ratio is high, alpha is high
    # When uncertainty_ratio is low, alpha is low
    # Use a smooth function
    alpha = np.clip(0.3 * np.sqrt(uncertainty_ratio), 0.0, 0.5)
    
    # Exploration score: beta * sigma
    exploration_score = beta * sigma_safe
    
    # Combine: weighted EI + weighted exploration
    scores = (1 - alpha) * ei + alpha * exploration_score
    
    # Additional: add a small boost to points with very high sigma
    # This helps escape local optima in the surrogate
    # Use a softmax-like boost on the top-k sigma values
    # But this might be too complex; let's keep it simple
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Make sure the best point (where EI is highest) gets a reasonable score
    # This should already be handled by the EI term
    
    return scores
# EVOLVE-BLOCK-END
