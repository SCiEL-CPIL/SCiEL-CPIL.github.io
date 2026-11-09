import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Probability of improvement (PI) - useful for early exploration
    pi = cdf_z
    
    # Upper Confidence Bound (UCB) for minimization
    # Adaptive beta: start high for exploration, decay as we find better solutions
    # f_best starts high (rastrigin min is 0, max ~40), so use normalized progress
    # When f_best is close to 0, we should exploit more
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)  # 0 when f_best=50, 1 when f_best=0
    beta = 2.0 * (1.0 - 0.5 * progress)  # beta ranges from 2.0 (early) to 1.0 (late)
    ucb = -mu + beta * sigma_safe
    
    # Weighted combination:
    # - EI: strong exploitation signal
    # - UCB: exploration-exploitation balance  
    # - PI: early exploration boost
    # 
    # Structure: blend EI and UCB with cost normalization
    # Add a small PI term to encourage exploration when improvement is likely
    w_ei = 1.0
    w_ucb = 0.5
    w_pi = 0.1 * (1.0 - progress)  # PI weight decays as we progress
    
    score = (w_ei * ei + w_ucb * ucb + w_pi * pi * sigma_safe) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
