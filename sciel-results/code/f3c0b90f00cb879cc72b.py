"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None

def _ensure_deadline():
    """Return True if we should continue, False if we've hit the safety deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    # Safety margin: return within 250s to be well under the 330s kill
    return (time.time() - _START_TIME) < 250.0

def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)

def _compute_ucb(mu, sigma, kappa=1.0):
    """Compute Upper Confidence Bound (for minimization, we want low mu + high sigma)."""
    return mu - kappa * sigma

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances Expected Improvement with evaluation cost,
    augmented with a UCB exploration term.
    
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
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Compute Expected Improvement
    ei = _compute_ei(mu, sigma, f_best)
    
    # Compute UCB (for minimization: lower is better, so we want mu - kappa*sigma)
    # But for acquisition score (higher = evaluate next), we want high values
    # UCB for minimization: score = -mu + kappa*sigma (higher means more promising)
    # Actually, let's think about this differently:
    # We want to select points where mu is low (good) and sigma is high (uncertain)
    # So a good score would be: -mu + kappa*sigma (higher is better)
    # But we also want to weight by cost
    
    # Combine EI and UCB-like term
    # EI already captures exploitation (high when mu < f_best)
    # UCB adds exploration (high when sigma is large)
    
    # Normalize EI and UCB components
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    # UCB score: -mu + kappa*sigma (higher = more promising for minimization)
    kappa = 1.0
    ucb_score = -mu + kappa * sigma
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < _EPS:
        ucb_range = 1.0
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weighted combination: 70% EI, 30% UCB
    # This balances exploitation and exploration
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Divide by cost to get cost-aware score
    scores = combined / cost
    
    # Check deadline
    if not _ensure_deadline():
        # If we're over deadline, return what we have
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
