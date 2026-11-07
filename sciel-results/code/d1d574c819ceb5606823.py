import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like exploration term
    to balance exploitation and exploration in the multi-modal Branin landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean at candidate points
    sigma : ndarray, shape (n,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero or near-zero sigma
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # --- Expected Improvement component ---
    Z = (f_best - mu) / sigma_safe
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    ei = np.maximum(ei, 0.0)
    
    # --- UCB exploration component ---
    # Normalize sigma to have similar scale as EI for blending
    # Use a moderate beta to encourage exploration of uncertain regions
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # --- Blend: use EI as primary, add exploration bonus for high-uncertainty low-mean regions ---
    # The EI already has some exploration via the sigma*phi(Z) term,
    # but we add an explicit UCB-like term to avoid getting trapped
    # in local optima of the GP mean.
    
    # Normalize both to comparable scales
    ei_mean = np.mean(ei)
    ucb_mean = np.mean(np.abs(ucb))
    
    # If EI is very small everywhere, rely more on UCB
    # Blend weight: 0.7 * EI + 0.3 * UCB (with sign correction)
    # UCB for minimization: lower mu + higher sigma is better
    # So we want high scores where mu is low and sigma is high
    
    # Scale EI and UCB to have similar magnitudes
    if ei_mean > 1e-10:
        ei_scaled = ei / ei_mean
    else:
        ei_scaled = ei
    
    if ucb_mean > 1e-10:
        ucb_scaled = ucb / ucb_mean
    else:
        ucb_scaled = ucb
    
    # Combine: primary exploitation via EI, secondary exploration via UCB
    # The UCB term helps escape local minima in the GP posterior mean
    scores = 0.7 * ei_scaled + 0.3 * ucb_scaled
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
