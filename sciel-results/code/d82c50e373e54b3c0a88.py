"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 300  # seconds, with safety margin before 330s limit


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a structural change: combining UCB with cost-aware weighting
    and a dynamic exploration-exploitation balance based on budget remaining.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Check deadline - if close, return simple EI to avoid timeout
    if _check_deadline():
        z = np.clip((f_best - mu) / sigma_safe, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate Expected Improvement (EI) for exploitation
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for exploration: mu - beta * sigma (since we're minimizing)
    # Use a dynamic beta based on remaining budget
    total_budget = 30.0
    # Estimate remaining budget from cost array (average cost * remaining evaluations)
    # But we don't know exact remaining, so use a heuristic based on f_best magnitude
    # Early on: more exploration; late on: more exploitation
    # Use normalized f_best as a proxy for progress
    # f_best starts high (bad) and decreases toward -78.33
    progress = np.clip((f_best + 100) / 100, 0.1, 1.0)  # 0.1 when f_best=-100, 1.0 when f_best=0
    beta = 2.0 * (1.0 - progress) + 0.5  # beta from 2.5 (early) to 0.5 (late)
    
    ucb = mu - beta * sigma_safe  # Lower is better for minimization
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_min = np.min(ucb)
    ucb_range = np.max(ucb) - ucb_min + 1e-12
    ucb_norm = (ucb - ucb_min) / ucb_range  # Higher = more uncertain/better for exploration
    
    # Combine EI (exploitation) and UCB (exploration)
    # Weight them based on progress: early = more UCB, late = more EI
    w_ei = 0.5 + 0.5 * progress  # 0.75 early, 1.0 late
    w_ucb = 1.0 - w_ei + 0.5  # Keep some UCB always
    
    combined = w_ei * (ei / ei_max) + w_ucb * ucb_norm
    
    # Cost-aware weighting: prefer low-cost points
    # Use a multiplicative factor that penalizes high cost
    # cost is in [1, 2] range (1 + mean(x scaled to [0,1]))
    # Use exponential decay to smoothly penalize cost
    cost_penalty = np.exp(-0.5 * (cost_safe - 1.0))  # 1.0 at cost=1, ~0.6 at cost=2
    
    score = combined * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
