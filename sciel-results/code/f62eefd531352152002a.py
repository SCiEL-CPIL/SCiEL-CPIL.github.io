"""Rewritten acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a structural change: adaptive exploration-exploitation
    with diversity-aware UCB and lower-bound confidence.
    
    Instead of pointwise EI or KG, this uses a hybrid of:
    1. UCB with adaptive beta that decreases as we get closer to the optimum
    2. A diversity bonus that rewards points far from already-evaluated regions
    3. A lower-bound term that accounts for the known minimum of ST-1D
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Known lower bound for ST-1D
    lb = -39.1662
    
    # Compute the gap between current best and known minimum
    gap = f_best - lb  # How much improvement is still possible
    
    # Normalize gap: if gap is large, we're early; if small, we're late
    # Use a smooth schedule based on gap
    gap_scale = max(gap, 0.01)
    progress = min(1.0, max(0.0, 1.0 - gap / 10.0))  # 0 early, 1 late
    
    # Adaptive UCB beta: start high (exploration), decrease toward exploitation
    # beta controls the tradeoff: score = -mu + beta * sigma
    # Higher beta = more exploration
    beta_base = 2.5 * (1.0 - 0.7 * progress) + 0.5 * progress
    # Ensure beta is in a reasonable range
    beta = np.clip(beta_base, 0.5, 3.0)
    
    # UCB score (we want to minimize, so higher UCB score = better exploration)
    ucb = -mu + beta * sigma
    
    # EI for reference
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1] range for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Diversity bonus: reward candidates with high sigma (uncertain regions)
    # This helps escape local optima by exploring uncharted territory
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        diversity = sigma / sigma_max
    else:
        diversity = np.zeros_like(sigma)
    
    # Lower-bound confidence: how confident are we that a point can beat lb?
    # P(f(x) < lb) = Phi((lb - mu) / sigma)
    z_lb = (lb - mu) / sigma
    p_beat_lb = norm.cdf(z_lb)
    # Only meaningful if there's a chance of beating the lower bound
    p_beat_lb = np.where(np.isfinite(p_beat_lb), p_beat_lb, 0.0)
    
    # Combine all terms with progress-aware weights
    # Early: weight diversity and UCB heavily
    # Late: weight EI and lower-bound confidence more
    
    w_ucb = 0.3 * (1.0 - progress) + 0.2 * progress
    w_ei = 0.3 * (1.0 - progress) + 0.4 * progress
    w_div = 0.25 * (1.0 - progress) + 0.1 * progress
    w_lb = 0.15 * (1.0 - progress) + 0.3 * progress
    
    # Normalize each component to comparable scale
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Diversity is already in [0, 1]
    # p_beat_lb is in [0, 1]
    
    # Final score: higher is better
    scores = w_ucb * ucb_norm + w_ei * ei_norm + w_div * diversity + w_lb * p_beat_lb
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
