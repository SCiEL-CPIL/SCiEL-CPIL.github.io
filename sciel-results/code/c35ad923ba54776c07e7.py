"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds total allowed (safety margin before 330s kill)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Combines Expected Improvement (EI) with a UCB-style exploration bonus.
    EI provides strong exploitation signal, while the exploration term
    helps escape local optima in the multimodal Styblinski-Tang landscape.
    
    The combination uses:
      - Normalized EI as the primary signal (weight ~0.7)
      - A UCB-style term (-mu + beta*sigma) normalized to [0,1] (weight ~0.3)
    
    This structural design leverages the proven effectiveness of EI for
    convergence while the UCB component provides a principled exploration
    mechanism that is scale-invariant.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: simple negative mean
        return -mu
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_normalized = (ei - ei_min) / ei_range
    else:
        ei_normalized = np.zeros_like(ei)
    
    # UCB component for minimization: -mu + beta * sigma
    # Use a moderate beta that balances exploration
    beta = 1.5
    ucb_raw = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1] range
    ucb_min = np.min(ucb_raw)
    ucb_max = np.max(ucb_raw)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_normalized = (ucb_raw - ucb_min) / ucb_range
    else:
        ucb_normalized = np.zeros_like(ucb_raw)
    
    # Combine: weighted sum of normalized EI and UCB
    # EI is the stronger signal for exploitation, UCB adds exploration
    ei_weight = 0.7
    ucb_weight = 0.3
    
    score = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
