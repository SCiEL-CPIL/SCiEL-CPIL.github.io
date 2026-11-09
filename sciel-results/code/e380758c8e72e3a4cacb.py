"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function with adaptive
    exploration bonus for Bayesian optimization of Hartmann 6D.
    
    PI scores candidates by the probability that f(x) < f_best, which is
    robust in high-dimensional spaces. An exploration bonus proportional to
    predictive uncertainty (sigma) is added to encourage sampling in
    under-explored regions, calibrated relative to the median uncertainty.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # For minimization, improvement means f(x) < f_best
    z = (f_best - mu) / sigma
    pi_scores = norm.cdf(z)
    
    # Exploration bonus: reward points with higher uncertainty than median
    # This helps escape local minima in the 6D Hartmann function
    sigma_median = np.median(sigma)
    if sigma_median < 1e-8:
        exploration_bonus = np.zeros_like(sigma)
    else:
        # Relative uncertainty above median, clipped to avoid extreme values
        rel_uncertainty = np.maximum((sigma - sigma_median) / sigma_median, 0.0)
        rel_uncertainty = np.minimum(rel_uncertainty, 3.0)  # cap at 3x median
        exploration_bonus = sigma_median * rel_uncertainty
    
    # Adaptive weighting: reduce exploration weight as optimization progresses
    # Use mean sigma as a proxy for remaining uncertainty / stage
    mean_sigma = np.mean(sigma)
    # When mean_sigma is high (early), we want more exploration
    # When mean_sigma is low (late), we want more exploitation
    # Map mean_sigma to a weight in [0.05, 0.3]
    # Assuming typical sigma range for standardized GP on Hartmann 6D
    # Early: mean_sigma ~ 0.5-1.0, Late: mean_sigma ~ 0.01-0.1
    # Use a smooth decay
    weight = 0.05 + 0.25 * np.exp(-mean_sigma * 5.0)
    weight = np.clip(weight, 0.05, 0.30)
    
    # Combine PI and exploration bonus
    # PI scores are in [0, 1], exploration_bonus is in objective units
    # Scale exploration bonus to be comparable to PI range
    # The PI range is at most 1.0, so we want the bonus to be a small fraction
    scores = pi_scores + weight * exploration_bonus / (1.0 + mean_sigma)
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
