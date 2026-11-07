"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute some constants for speed
_EPS = 1e-12

def _expected_improvement(mu, sigma, f_best):
    """Compute standard Expected Improvement (EI)."""
    z = (f_best - mu) / (sigma + _EPS)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb_score(mu, sigma, f_best, beta):
    """Compute UCB score. Note: we are minimizing, so we want low mu and high sigma."""
    return -mu + beta * sigma

def _qei_score(mu, sigma, f_best, xi):
    """Quantitative Expected Improvement."""
    z = (f_best - mu - xi) / (sigma + _EPS)
    qei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(qei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of GP at candidate points.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive to avoid division by zero
    cost = np.maximum(cost, _EPS)
    
    # Compute multiple acquisition functions and combine them
    # 1. Cost-aware EI
    ei = _expected_improvement(mu, sigma, f_best)
    ca_ei = ei / cost
    
    # 2. Cost-aware UCB (beta tuned for exploration)
    # Use a moderate beta value
    beta = 2.0
    ucb = _ucb_score(mu, sigma, f_best, beta)
    ca_ucb = ucb / cost
    
    # 3. Cost-aware QEI (with small xi)
    xi = 0.01 * max(abs(f_best), 1e-6)
    qei = _qei_score(mu, sigma, f_best, xi)
    ca_qei = qei / cost
    
    # Combine: weighted sum of the three cost-aware scores
    # Weights chosen to balance exploitation (EI, QEI) and exploration (UCB)
    w_ei = 0.4
    w_ucb = 0.3
    w_qei = 0.3
    
    scores = w_ei * ca_ei + w_ucb * ca_ucb + w_qei * ca_qei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
