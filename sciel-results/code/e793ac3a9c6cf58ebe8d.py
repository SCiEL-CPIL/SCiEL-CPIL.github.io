"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time to ensure we return within budget.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a structurally different approach: a multi-objective acquisition
    combining Expected Improvement (EI) with a diversity-aware exploration
    term based on posterior variance ranking. This avoids the fixed UCB
    trade-off and instead adapts exploration based on the current posterior
    structure.
    
    Key structural difference: Instead of a static EI + UCB blend, we use
    a rank-based diversity bonus that rewards candidates in high-uncertainty
    regions that are NOT already well-covered by the posterior mean landscape.
    This is inspired by knowledge gradient ideas but computed efficiently.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        # Return a safe fallback: just negative mu (exploitation only)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        scores = -mu + 10.0 * sigma
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- Structural change: Rank-based diversity exploration ---
    # Instead of a fixed UCB, we compute a diversity score based on
    # how "unique" each candidate's posterior prediction is.
    # We use the posterior variance (sigma^2) as a proxy for information
    # gain, but weight it by how far the mean is from the best,
    # to avoid pure exploration in already-good regions.
    
    # Information gain proxy: sigma^2 * (1 - norm.cdf(z))
    # This captures regions with high uncertainty that have a chance of improvement
    info_gain = sigma_safe ** 2 * (1.0 - norm.cdf(z))
    
    # Normalize EI and info_gain to [0, 1] for fair combination
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    ig_max = np.max(info_gain)
    ig_min = np.min(info_gain)
    if ig_max - ig_min > 1e-12:
        ig_norm = (info_gain - ig_min) / (ig_max - ig_min)
    else:
        ig_norm = np.zeros_like(info_gain)
    
    # Adaptive weighting: more exploration early (when sigma is large relative to problem scale)
    # Problem scale: goldstein price ranges roughly [3, 40] on [-2,2]^2
    # Use the median sigma as a proxy for current exploration level
    med_sigma = np.median(sigma_safe)
    
    # If median sigma is large, we're early -> explore more
    # If median sigma is small, we're late -> exploit more
    # Scale: sigma ~ 5 means early, sigma ~ 0.5 means late
    exploration_weight = np.clip(med_sigma / 3.0, 0.2, 0.8)
    
    # Combine: exploitation (EI) + exploration (information gain)
    scores = (1.0 - exploration_weight) * ei_norm + exploration_weight * ig_norm
    
    # Add a small bonus for candidates with very low predicted values
    # This helps break ties and pushes toward promising regions
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-12:
        mu_promising = (mu_max - mu) / (mu_max - mu_min)
    else:
        mu_promising = np.zeros_like(mu)
    
    scores += 0.1 * mu_promising
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
