"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None
_DEADLINE = 280.0  # safety margin before 330s kill


def _ensure_deadline():
    """Return True if we should continue, False if we've hit the safety deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < _DEADLINE


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def _compute_pi(mu, sigma, f_best):
    """Compute Probability of Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    return np.maximum(pi, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using a cost-aware Bayesian Optimization under Budget (BOB) approach.
    Maximizes EI per unit cost with an adaptive exploration component.
    
    Args:
        mu: numpy array of shape (2000,), predicted means
        sigma: numpy array of shape (2000,), predicted standard deviations
        f_best: float, best observed value so far
        cost: numpy array of shape (2000,), evaluation costs for each candidate
        
    Returns:
        numpy array of shape (2000,), acquisition scores (higher is better)
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Convert to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute Expected Improvement
    ei = _compute_ei(mu, sigma, f_best)
    
    # Compute Probability of Improvement
    pi = _compute_pi(mu, sigma, f_best)
    
    # Cost-aware EI: EI divided by cost (marginal improvement per budget unit)
    ei_per_cost = ei / cost
    
    # Adaptive exploration: UCB-like term with cost normalization
    # kappa controls exploration; we use a quadratic cost penalty to
    # create sharper preference for cheap uncertain regions
    kappa = 2.0
    ucb = mu + kappa * sigma
    # Cost-normalized UCB: penalize expensive points quadratically
    cost_penalty = 1.0 + 0.5 * (cost - 1.0) ** 2
    ucb_cost_aware = ucb / cost_penalty
    
    # Normalize EI/cost to [0, 1]
    ei_pc_max = np.max(ei_per_cost)
    if ei_pc_max < _EPS:
        ei_pc_max = 1.0
    ei_pc_norm = ei_per_cost / ei_pc_max
    
    # Normalize UCB/cost to [0, 1]
    ucb_ca_max = np.max(ucb_cost_aware)
    if ucb_ca_max < _EPS:
        ucb_ca_max = 1.0
    ucb_ca_norm = ucb_cost_aware / ucb_ca_max
    
    # Blend: 60% EI/cost (exploitation), 40% cost-aware UCB (exploration)
    scores = 0.6 * ei_pc_norm + 0.4 * ucb_ca_norm
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
