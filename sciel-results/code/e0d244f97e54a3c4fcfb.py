import numpy as np
import time
from scipy.special import erf

# Global deadline guard: 330s wall-clock limit, use 320s as safety margin
_start_time = time.time()
_DEADLINE = _start_time + 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Combines Expected Improvement (EI) for exploitation with a deterministic
    Upper Confidence Bound (UCB) term for explicit global exploration.
    The UCB term targets regions of high posterior uncertainty to handle
    the multimodal nature of the Weierstrass function, while EI focuses
    on improving the current best.
    
    Args:
        mu: array of shape (2000,) - posterior mean at candidates
        sigma: array of shape (2000,) - posterior std at candidates
        f_best: float - best observed function value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Deadline guard: if we're too close to the limit, use a fast fallback
    if time.time() > _DEADLINE - 5.0:
        # Fast fallback: simple LCB for minimization
        return -(mu - 0.5 * sigma_safe)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical overflow in exp
    z_clipped = np.clip(z, -50, 50)
    
    # Compute Phi(z) and phi(z) manually for speed
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / sqrt_2pi
    Phi_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Deterministic UCB for minimization
    # We want to maximize the upper bound of potential improvement
    # UCB_score = -(mu + kappa * sigma) for minimization
    # But we want to combine with EI, so let's use a different formulation:
    # The UCB term should reward high uncertainty regions
    
    # Adaptive kappa: use a fixed moderate value for global exploration
    # The Weierstrass function has many local minima, so we need good exploration
    kappa = 1.5
    
    # UCB score for minimization: higher is better
    # We want to explore where mu is low AND sigma is high
    # ucb_score = -mu + kappa * sigma (this is LCB for minimization, but we call it UCB
    # in the sense that it's an upper bound on potential improvement)
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize both scores to zero mean, unit variance for fair combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        z_ei = np.zeros_like(ei)
    else:
        z_ei = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std < 1e-9:
        z_ucb = np.zeros_like(ucb_score)
    else:
        z_ucb = (ucb_score - ucb_mean) / ucb_std
    
    # Combine EI and UCB with adaptive weights
    # Early on (high uncertainty), weight UCB more for exploration
    # Later (low uncertainty), weight EI more for exploitation
    
    # Use mean sigma as proxy for exploration level
    mean_sigma = np.mean(sigma_safe)
    # Normalize: if mean_sigma is high, we're early; if low, we're late
    # Typical sigma range for this problem is roughly [0.01, 0.5]
    sigma_norm = np.clip(mean_sigma / 0.3, 0.0, 1.0)
    
    # Weights: early = more UCB, late = more EI
    w_ucb = 0.3 + 0.4 * sigma_norm  # ranges from 0.3 to 0.7
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * z_ei + w_ucb * z_ucb
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
