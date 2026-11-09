"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structural approach: Thompson Sampling with cost-awareness,
    combined with a multi-objective balancing of exploitation and exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Time check - if close to timeout, return a simple but valid score
    if not _elapsed_ok():
        # Fallback: simple cost-normalized improvement
        improvement = f_best - mu
        scores = np.maximum(improvement, 0.0) / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Core approach: Thompson Sampling with cost awareness
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # We want to minimize, so lower samples are better
    # But we need deterministic scores, so use expected value of a cost-adjusted objective
    
    # Strategy: combine multiple acquisition perspectives
    
    # 1. Expected Improvement (exploitation)
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # 2. Probability of Improvement (exploration of uncertain regions)
    poi = norm.cdf(z)
    
    # 3. Uncertainty-based exploration
    # Use a scaled uncertainty term that accounts for the scale of the problem
    mu_scale = np.std(mu) + 1e-9
    sigma_normalized = sigma / mu_scale
    
    # 4. Cost-aware Thompson Sampling
    # Sample theta from N(f_best, sigma^2) for each candidate
    # The expected cost-adjusted improvement is what we want to maximize
    
    # For Thompson Sampling, we want to maximize the probability that
    # a sample from the posterior is better than f_best, weighted by cost
    
    # Effective score: E[max(f_best - f(x), 0)] / cost, but we approximate this
    # with a combination of EI and exploration
    
    # Adaptive exploration rate based on the current state
    # When we're close to the optimum (small EI values), explore more
    avg_ei = np.mean(ei)
    max_ei = np.max(ei)
    
    # Exploration coefficient: higher when exploitation is weak
    if max_ei > 1e-9:
        # When EI is large, we should exploit more
        exploration_weight = 0.3 + 0.7 * (1.0 - avg_ei / (max_ei + 1e-9))
    else:
        exploration_weight = 1.0
    
    exploration_weight = np.clip(exploration_weight, 0.1, 1.0)
    
    # Combined raw score before cost adjustment
    # Balance between EI (exploitation) and uncertainty (exploration)
    raw_score = ei + exploration_weight * sigma_normalized * mu_scale
    
    # Cost penalty: use a power law to penalize expensive evaluations
    # gamma > 1 makes cost matter more, gamma < 1 makes it matter less
    # For this problem, cost varies from 1 to 2, so we want moderate sensitivity
    gamma = 1.2
    cost_factor = cost ** gamma
    
    scores = raw_score / cost_factor
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if not _elapsed_ok():
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
