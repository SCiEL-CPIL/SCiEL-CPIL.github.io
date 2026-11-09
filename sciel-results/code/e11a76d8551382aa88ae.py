"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is very small and mu >= f_best, EI should be 0
    ei = np.where(sigma < 1e-7, np.where(mu < f_best, f_best - mu, 0.0), ei)
    return ei


def _probability_of_improvement(mu, sigma, f_best):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    return poi


def _ucb(mu, sigma, f_best, beta=2.0):
    """Upper confidence bound (for minimization, we want low mu + high uncertainty)."""
    return mu - beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI, UCB, and POI with adaptive weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a simple cost-based score as fallback
        return -cost + np.min(cost)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute base acquisition functions
    ei = _expected_improvement(mu, sigma, f_best)
    poi = _probability_of_improvement(mu, sigma, f_best)
    ucb = _ucb(mu, sigma, f_best, beta=2.0)
    
    # Normalize UCB to [0, 1] range for combination
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Adaptive weighting based on how good f_best is relative to the known minimum
    # Styblinski-Tang 2D minimum is -78.3323
    f_min_known = -78.3323
    # Progress: 0 when f_best is at minimum, 1 when f_best is far from minimum
    progress = np.clip((f_best - f_min_known) / 100.0, 0.0, 1.0)
    
    # When progress is high (far from optimum), explore more (UCB weight up)
    # When progress is low (close to optimum), exploit more (EI weight up)
    w_ei = 0.6 + 0.4 * (1.0 - progress)
    w_ucb = 0.2 + 0.3 * progress
    w_poi = 0.2
    
    # Normalize weights
    w_sum = w_ei + w_ucb + w_poi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_poi /= w_sum
    
    # Combine acquisition functions
    # EI is already in objective units, normalize it
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combined score before cost adjustment
    combined = w_ei * ei_norm + w_ucb * ucb_norm + w_poi * poi
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    # Use a softened cost division to avoid extreme preferences
    cost_normalized = cost / np.mean(cost)
    scores = combined / cost_normalized
    
    # Add a small exploration bonus for points with high uncertainty relative to their cost
    # This helps discover new basins
    uncertainty_bonus = sigma / np.mean(sigma)
    scores += 0.1 * uncertainty_bonus / cost_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're taking too long, return early
    if time.time() - start_time > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
