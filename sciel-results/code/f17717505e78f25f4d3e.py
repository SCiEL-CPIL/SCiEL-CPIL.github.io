"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
_EPS = 1e-9
_CDF_PDF_CACHE = None

def _compute_hybrid_acquisition(mu, sigma, f_best, cost):
    """
    Compute a hybrid acquisition score combining EI and UCB, both cost-normalized.
    Uses an adaptive beta that decreases as remaining budget shrinks, shifting
    from exploration to exploitation.
    """
    # Expected Improvement
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB: mu + beta * sigma
    # Adaptive beta: start with moderate exploration, decrease as cost accumulates
    # cost is in [1, 2] range (1 + mean(x_scaled)), total budget is 30
    # We don't know total spent, but we can use the mean cost as a proxy
    # for "how far along" we are. Lower mean cost of candidates suggests
    # we're in regions we haven't explored much.
    mean_cost = np.mean(cost)
    # Beta starts at ~1.5 and decreases toward ~0.3 as average cost increases
    # (higher cost regions are more expensive, so we should explore less there)
    beta = 1.5 - 1.2 * (mean_cost - 1.0)  # mean_cost in [1,2], so beta in [0.3, 1.5]
    beta = np.maximum(beta, 0.1)
    
    ucb = mu + beta * sigma
    
    # Cost-normalize both terms
    ei_per_cost = ei / cost
    ucb_per_cost = ucb / cost
    
    # Combine: 0.7 * EI/cost + 0.3 * UCB/cost
    # The UCB term helps when EI is small but uncertainty is high
    scores = 0.7 * ei_per_cost + 0.3 * ucb_per_cost
    
    return scores

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances Expected Improvement with evaluation cost.
    
    Args:
        mu: numpy array of shape (2000,), predicted means
        sigma: numpy array of shape (2000,), predicted standard deviations
        f_best: float, best observed value so far
        cost: numpy array of shape (2000,), evaluation costs for each candidate
        
    Returns:
        numpy array of shape (2000,), acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin: return within 5s to be safe
    
    # Convert to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Compute hybrid acquisition score
    scores = _compute_hybrid_acquisition(mu, sigma, f_best, cost)
    
    # Check deadline (should be fast, but just in case)
    if time.time() > deadline:
        return scores
    
    # The scores should be finite. Replace any non-finite values with a very low score.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
