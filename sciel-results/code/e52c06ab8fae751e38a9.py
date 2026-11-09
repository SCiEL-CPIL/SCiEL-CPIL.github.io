import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for deadline
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    
    # Safety for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score for EI computation
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    EI = sigma_safe * (z * Phi + phi)
    EI = np.maximum(EI, 0.0)
    
    # Compute UCB with adaptive beta based on the scale of uncertainty
    # Use a data-driven exploration parameter
    # beta scales with the relative spread of sigma to balance exploration
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    
    # Adaptive beta: larger when uncertainty is high relative to best
    # This encourages exploration early, exploitation later
    if sigma_mean > 1e-10:
        # Normalize by the range of mu to make it scale-invariant
        mu_range = np.max(mu) - np.min(mu) if np.max(mu) > np.min(mu) else 1.0
        beta = 0.5 * (sigma_mean / (abs(f_best) + mu_range + 1e-10))
        beta = np.clip(beta, 0.1, 2.0)
    else:
        beta = 0.5
    
    # UCB: mu + beta * sigma (for minimization, we want low mu, so we use -mu + beta*sigma)
    # Actually for minimization, UCB acquisition = -mu + beta * sigma
    UCB = -mu + beta * sigma_safe
    
    # Cost-normalized scores
    EI_score = EI / cost_safe
    UCB_score = UCB / cost_safe
    
    # Combine: weighted sum with adaptive weighting
    # Weight EI more when we're close to the optimum, UCB more when exploring
    # Use the ratio of EI to UCB as a signal
    total_ei = np.sum(EI_score)
    total_ucb = np.sum(np.maximum(UCB_score, 0))
    
    if total_ei + total_ucb > 1e-10:
        w_ei = total_ei / (total_ei + total_ucb)
        w_ucb = 1.0 - w_ei
    else:
        w_ei = 0.5
        w_ucb = 0.5
    
    # Additional exploration bonus for points with high sigma relative to cost
    # This helps escape local optima
    sigma_cost_ratio = sigma_safe / cost_safe
    sigma_cost_max = np.max(sigma_cost_ratio) if sigma_cost_max > 0 else 1.0
    exploration_bonus = 0.3 * (sigma_cost_ratio / (sigma_cost_max + 1e-10))
    
    # Final score
    score = w_ei * EI_score + w_ucb * UCB_score + exploration_bonus
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
