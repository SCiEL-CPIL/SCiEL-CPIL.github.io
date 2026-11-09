"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Dynamic acquisition function that adapts between UCB and EI based on
    the posterior sharpness metric.
    
    Parameters:
    mu: array-like, shape (N,), posterior mean
    sigma: array-like, shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array-like, shape (N,), higher scores indicate better candidates
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute a sharpness metric: how concentrated the posterior is
    # Lower values mean the GP is more certain (exploit more)
    # Higher values mean the GP is uncertain (explore more)
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Normalized sharpness: high when sigma is small and uniform (exploit)
    # Low when sigma is large or varies (explore)
    if sigma_mean > 1e-6:
        sharpness = 1.0 / (sigma_mean + 1e-6)
    else:
        sharpness = 0.0
    
    # Dynamic kappa: starts high for exploration, decreases as we get more confident
    # Use a sigmoid-like transition based on sharpness
    # When sharpness is low (high uncertainty), kappa is large (explore)
    # When sharpness is high (low uncertainty), kappa is small (exploit)
    base_kappa = 3.0
    min_kappa = 0.5
    
    # Transition from exploration to exploitation
    # sigmoid(5 * (sharpness - 1)) maps sharpness ~1 to 0.5
    # For sharpness < 0.5, sigmoid is small -> kappa close to base
    # For sharpness > 1.5, sigmoid is large -> kappa close to min
    transition = 1.0 / (1.0 + np.exp(-5.0 * (sharpness - 1.0)))
    kappa = base_kappa - (base_kappa - min_kappa) * transition
    
    # Also compute EI for potential blending
    # EI for minimization: E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB score for minimization
    ucb_score = -mu + kappa * sigma
    
    # Blend between UCB and EI based on sharpness
    # When uncertain (low sharpness), rely more on UCB (exploration)
    # When certain (high sharpness), rely more on EI (exploitation)
    blend_weight = transition  # 0 when exploring, 1 when exploiting
    
    scores = (1 - blend_weight) * ucb_score + blend_weight * ei
    
    return scores
# EVOLVE-BLOCK-END
