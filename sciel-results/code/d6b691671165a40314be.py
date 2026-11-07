"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with a schedule that shifts
    from exploration to exploitation as f_best approaches 0.
    
    mu, sigma: arrays of shape (N,) in objective units
    f_best: lowest observed value so far
    Returns: array of shape (N,) with higher scores meaning 'evaluate next'
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin well under 300s per eval

    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-12)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)

    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive UCB: beta decreases as f_best gets closer to 0 (the minimum)
    # f_best is in objective units; minimum is 0.
    # When f_best is large (early), we explore more; when f_best ~ 0, we exploit.
    # Scale: f_best / (median_sigma) gives a dimensionless measure of progress
    med_sigma = np.median(sigma)
    if med_sigma > 1e-12:
        progress = np.abs(f_best) / med_sigma
    else:
        progress = 1.0
    
    # Beta: start high for exploration, decay as we get close to optimum
    # Use a smooth decay: beta = beta_max * exp(-progress / tau)
    beta_max = 2.5
    tau = 3.0  # Time constant for decay
    beta = beta_max * np.exp(-progress / tau)
    
    # Add a floor to beta to maintain some exploration
    beta = max(beta, 0.1)

    # UCB term
    ucb = mu + beta * sigma

    # Combine EI and UCB: weight them adaptively
    # When progress is high (f_best close to 0), rely more on EI (exploitation)
    # When progress is low, rely more on UCB (exploration)
    # weight_ei goes from 0.3 (early) to 0.9 (late)
    weight_ei = 0.3 + 0.6 * (1.0 - np.exp(-progress / tau))
    weight_ucb = 1.0 - weight_ei

    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-12)

    score = weight_ei * ei_norm + weight_ucb * ucb_norm

    # Check deadline
    if time.time() > deadline:
        return score

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
