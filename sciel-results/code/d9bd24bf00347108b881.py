import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid acquisition: UCB with time-decaying beta + EI.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Estimate how many observations have been made using median sigma.
    # With standardized targets and Matern-5/2, sigma starts near 1.0
    # and decreases as more data is collected.
    med_sigma = np.median(sigma_safe)
    
    # Map median sigma to a "learning progress" in [0, 1].
    # sigma ~ 1.0 means early (few obs), sigma ~ 0.1 means late (many obs).
    # Use a smooth mapping: progress = 1 - (med_sigma / 1.0) clipped.
    progress = np.clip(1.0 - med_sigma, 0.0, 1.0)
    
    # Beta decays from ~3.0 (heavy exploration) to ~0.3 (exploitation).
    # Use exponential decay for smooth transition.
    beta_max = 3.0
    beta_min = 0.3
    beta = beta_min + (beta_max - beta_min) * np.exp(-4.0 * progress)
    
    # UCB component: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine: weight EI more as we learn more, UCB more early on.
    # w_ei goes from 0.2 (early) to 0.8 (late)
    w_ei = 0.2 + 0.6 * progress
    w_ucb = 1.0 - w_ei
    
    # Normalize EI to be comparable in scale to UCB.
    # EI is typically smaller in magnitude; scale it up.
    ei_scale = np.max(np.abs(ei)) + 1e-10
    ucb_scale = np.max(np.abs(ucb)) + 1e-10
    ei_norm = ei / ei_scale
    ucb_norm = ucb / ucb_scale
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
