"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Strategy:
    EI is the expected value of the improvement over the current best f_best.
    For minimization, EI(x) = E[max(f_best - f(x), 0)].
    
    Formula:
    Let Z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    Where:
    - mu is the predicted mean
    - sigma is the predicted standard deviation
    - f_best is the current minimum observed value
    - Phi is the CDF of the standard normal distribution
    - phi is the PDF of the standard normal distribution
    
    This approach is materially different from UCB because it is a probabilistic
    expectation of improvement, not a heuristic bound. It naturally handles the
    trade-off:
    - If sigma is small and mu < f_best, EI is large (exploitation).
    - If sigma is large, EI increases but is bounded by the uncertainty, 
      preventing excessive exploration in areas with no chance of improvement.
    
    We add a small exploitation constant xi to ensure we prioritize points 
    slightly better than f_best, which helps in the "polishing" phase.
    """
    global _start_time
    
    # Initialize deadline if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline: if we've used > 300s, return a safe fallback
    # The evaluator kills at 330s. We want to finish well before.
    # However, EI is vectorized and very fast (O(N)), so this is unlikely to trigger
    # unless the input sizes are huge, but the contract says 2000 points.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just return -mu to exploit the best current guess
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploitation constant. 
    # A small xi encourages finding points that are *better* than f_best, 
    # not just equal. For Beale, where min is 0, this helps descend the valley.
    xi = 1e-4
    
    # Calculate Z-score for improvement
    # Z = (f_best - mu) / sigma
    # If f_best - mu > 0, we are in a region where the mean is already better than best.
    # If f_best - mu < 0, we need the tail probability of sigma to overcome the deficit.
    Z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If (f_best - mu) is negative, Phi(Z) is small, but sigma*phi(Z) might still contribute.
    # This formula is standard for minimization EI.
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Add the exploitation constant term? 
    # Standard EI with xi: EI_xi = (f_best - mu - xi) * Phi(Z_xi) + sigma * phi(Z_xi)
    # where Z_xi = (f_best - mu - xi) / sigma
    # Let's use the xi version for better polishing behavior.
    
    Z_xi = (f_best - mu - xi) / sigma_safe
    Phi_xi = norm.cdf(Z_xi)
    phi_xi = norm.pdf(Z_xi)
    
    ei_xi = (f_best - mu - xi) * Phi_xi + sigma_safe * phi_xi
    
    # Ensure finiteness
    ei_xi = np.where(np.isfinite(ei_xi), ei_xi, 0.0)
    
    return ei_xi
# EVOLVE-BLOCK-END
