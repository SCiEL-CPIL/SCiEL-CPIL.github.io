"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for tracking number of evaluations
_eval_count = 0


def _smooth_sigma(sigma, kernel_width=0.1):
    """
    Apply a Gaussian kernel to smooth the sigma values spatially.
    Since we don't have coordinates, we assume the 2000 points are ordered
    in a structured grid or use index-based smoothing as a proxy for spatial smoothing.
    
    For a 2000-point set, we'll use a simple 1D Gaussian smoothing along the index,
    which serves as a proxy for spatial smoothing when coordinates aren't available.
    """
    n = len(sigma)
    if n <= 1:
        return sigma.copy()
    
    # Create index array
    idx = np.arange(n)
    
    # Gaussian kernel weights
    # Use a moderate kernel width relative to the number of points
    # This creates a smoothed version of sigma
    kernel_width = max(1, int(0.05 * n))  # 5% of points as kernel width
    
    # Efficient convolution using FFT
    kernel = np.exp(-0.5 * (idx - n // 2) ** 2 / (kernel_width ** 2))
    kernel /= np.sum(kernel)
    
    # Use FFT-based convolution for efficiency
    from scipy.signal import fftconvolve
    smoothed = fftconvolve(sigma, kernel, mode='same')
    
    return smoothed


def acquisition(mu, sigma, f_best):
    """
    UCB acquisition function with dynamically adjusted beta.
    
    Uses a Gaussian-smoothed sigma to create a spatially smoothed uncertainty map,
    then applies UCB with beta that decays exponentially with the number of evaluations.
    
    Early in the search: high beta -> aggressive exploration
    Late in the search: low beta -> local exploitation
    """
    global _eval_count
    
    t0 = time.time()
    
    # Increment evaluation counter
    _eval_count += 1
    
    # Total budget is 30 evaluations, 5 initial + 25 optimization steps
    # We're in the optimization phase after the first 5
    # So _eval_count here represents the step in the optimization loop
    # Step 1 corresponds to evaluation 6, step 25 corresponds to evaluation 30
    total_opt_steps = 25
    current_step = min(_eval_count, total_opt_steps)
    
    # Dynamically adjust beta: starts high, decays exponentially
    # beta = beta_max * exp(-decay_rate * step)
    beta_max = 5.0
    decay_rate = 0.15
    beta = beta_max * np.exp(-decay_rate * current_step)
    
    # Ensure beta doesn't go below a minimum to maintain some exploration
    beta_min = 0.1
    beta = max(beta, beta_min)
    
    # Smooth the sigma to create a spatially smoothed uncertainty map
    sigma_smoothed = _smooth_sigma(sigma)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score means more promising (lower expected value or high uncertainty)
    scores = -mu + beta * sigma_smoothed
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin before 300s limit
        # Fall back to simple exploitation (negative mean)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
