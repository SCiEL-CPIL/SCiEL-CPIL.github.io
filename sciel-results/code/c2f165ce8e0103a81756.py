"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Variance-Gated Expected Improvement (VEI) that scales EI by
    a power of relative sigma to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    t0 = time.time()
    deadline = t0 + 300.0
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    
    # Phi and phi
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI for minimization
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Variance-Gated EI: scale by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max < eps:
        sigma_ratio = np.ones_like(sigma_safe)
    else:
        sigma_ratio = sigma_safe / sigma_max
    
    gamma = 1.5
    gate = sigma_ratio ** gamma
    
    # VEI = EI * gate
    VEI = EI * gate
    
    # Also compute a pure UCB term for additional exploration
    # LCB score = -mu + beta * sigma (maximize this)
    beta = 2.0
    UCB = beta * sigma_safe - mu
    
    # Normalize both to [0, 1]
    VEI_max = np.max(VEI)
    VEI_min = np.min(VEI)
    VEI_range = VEI_max - VEI_min
    if VEI_range < eps:
        VEI_norm = np.ones_like(VEI) * 0.5
    else:
        VEI_norm = (VEI - VEI_min) / VEI_range
    
    UCB_max = np.max(UCB)
    UCB_min = np.min(UCB)
    UCB_range = UCB_max - UCB_min
    if UCB_range < eps:
        UCB_norm = np.ones_like(UCB) * 0.5
    else:
        UCB_norm = (UCB - UCB_min) / UCB_range
    
    # Blend: 70% VEI, 30% UCB
    # VEI provides exploitation with uncertainty-aware gating
    # UCB provides pure exploration
    blend_weight_vei = 0.7
    blend_weight_ucb = 0.3
    
    scores = blend_weight_vei * VEI_norm + blend_weight_ucb * UCB_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
