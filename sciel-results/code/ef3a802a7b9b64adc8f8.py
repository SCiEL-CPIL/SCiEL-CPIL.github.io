"""Cost-aware acquisition for Bayesian optimization on Powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = time.time()
DEADLINE = 300.0
SAFETY_MARGIN = 10.0


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using dynamic beta shifting.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty (sigma) for exploration
    3. Cost-aware normalization
    4. Dynamic beta that shifts from exploration to exploitation
       based on optimization progress (f_best state)
    """
    # Deadline guard
    if _elapsed() > DEADLINE - SAFETY_MARGIN:
        return np.ones(2000, dtype=np.float64)
    
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size != 2000 or sigma.size != 2000 or cost.size != 2000:
        return np.ones(2000, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic beta based on optimization progress
    # As f_best decreases (gets closer to 0), we shift toward exploitation
    # Use a sigmoid-like mapping: when f_best is high (early), beta is high (more exploration)
    # When f_best is low (late), beta is low (more exploitation)
    
    # Powell 4d minimum is 0, so f_best approaching 0 means we're close to optimal
    # Scale: if f_best > 10, we're early; if f_best < 1, we're late
    # Use tanh for smooth transition
    f_best_scaled = np.tanh((f_best - 5.0) / 5.0)  # ranges from -1 to 1
    # Map to beta: early (f_best high) -> beta ~ 1.0, late (f_best low) -> beta ~ 0.3
    beta = 0.3 + 0.7 * (f_best_scaled + 1.0) / 2.0
    beta = np.clip(beta, 0.2, 1.5)
    
    # Cost-aware acquisition: combine EI and uncertainty with dynamic beta
    # ei_term: exploitation signal
    ei_term = ei / cost_safe
    
    # exploration_term: uncertainty signal
    exploration_term = beta * sigma_safe / cost_safe
    
    # Combine
    scores = ei_term + exploration_term
    
    # Handle NaN/Inf values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > DEADLINE - SAFETY_MARGIN:
        pass
    
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
