"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return pure exploitation to finish quickly
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Risk-sensitive LCB using NLL-derived exploration
    # The key idea: use a non-linear, heteroscedastic exploration term
    # that allocates more exploration where uncertainty is high relative to current best
    
    # Compute z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # NLL-based acquisition: -log likelihood of observing f_best given the GP
    # For a Gaussian with mean mu and std sigma, the log-likelihood of f_best is:
    # -0.5 * log(2*pi*sigma^2) - 0.5 * ((f_best - mu)/sigma)^2
    # We want to maximize this (minimize negative log-likelihood)
    # But we also want to balance exploration and exploitation
    
    # A more principled approach: use the expected improvement with a risk-averse twist
    # Standard EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    from scipy.stats import norm
    
    # Compute standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus that is non-linear in sigma
    # The NLL insight: regions with high sigma contribute more to uncertainty
    # Use a softplus-like transformation to make exploration non-linear
    # exploration_bonus = alpha * log(1 + sigma_safe / scale)
    
    # Adaptive exploration: higher in early stages, lower later
    if _call_count <= 3:
        alpha = 0.3
        beta = 0.5
    elif _call_count <= 6:
        alpha = 0.2
        beta = 0.3
    elif _call_count <= 10:
        alpha = 0.1
        beta = 0.2
    elif _call_count <= 15:
        alpha = 0.05
        beta = 0.1
    else:
        alpha = 0.02
        beta = 0.05
    
    # Non-linear exploration term using log(1 + sigma/scale)
    # This gives diminishing returns for very high sigma
    scale = 0.1  # scale parameter for the log transformation
    exploration_bonus = alpha * np.log1p(sigma_safe / scale)
    
    # Combine EI with exploration bonus
    scores = ei + exploration_bonus
    
    # Add a small UCB component for additional exploration
    ucb_component = beta * sigma_safe
    
    scores = scores + ucb_component
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
