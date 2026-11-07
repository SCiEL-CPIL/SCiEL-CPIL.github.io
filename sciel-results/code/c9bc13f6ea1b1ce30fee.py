"""Cost-aware acquisition function combining EI with exploration bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _remaining():
    return 320.0 - (time.time() - _get_start_time())

_EPS = 1e-9

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus based on predictive uncertainty.
    """
    _get_start_time()
    
    n = len(mu)
    
    # Check if we're running low on time
    if _remaining() < 0.1:
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.maximum(cost, _EPS)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Compute base EI (Expected Improvement)
    z_base = (f_best - mu) / sigma_safe
    ei_base = (f_best - mu) * norm.cdf(z_base) + sigma * norm.pdf(z_base)
    ei_base = np.maximum(ei_base, 0.0)
    
    # Compute UCB-style exploration score
    # UCB = mu - kappa * sigma (for minimization, we want low mu and high sigma)
    # Actually for minimization: acquisition = mu - kappa * sigma is not right
    # For minimization, we want to minimize, so lower mu is better
    # UCB for minimization: score = mu - kappa * sigma doesn't make sense directly
    # Let's think: we want high acquisition scores for points we expect to be good
    # For minimization, good points have low f(x)
    # EI already captures this: high EI means expected improvement over f_best
    # The exploration bonus should reward points with high uncertainty (high sigma)
    
    # Let's use a combination:
    # score = EI / cost + beta * sigma / cost
    # This rewards both expected improvement and exploration
    
    # Try several parameter settings
    best_scores = None
    best_proxy = -np.inf
    
    # Parameters: (beta, cost_power)
    # beta controls exploration weight
    # cost_power controls how aggressively we penalize cost
    param_candidates = [
        (0.0, 1.0),    # Pure EI / cost
        (0.1, 1.0),    # EI/cost + 0.1*sigma/cost
        (0.5, 1.0),    # EI/cost + 0.5*sigma/cost
        (1.0, 1.0),    # EI/cost + 1.0*sigma/cost
        (0.1, 0.5),    # EI/sqrt(cost) + 0.1*sigma/sqrt(cost)
        (0.5, 0.5),    # EI/sqrt(cost) + 0.5*sigma/sqrt(cost)
        (1.0, 0.5),    # EI/sqrt(cost) + 1.0*sigma/sqrt(cost)
        (0.1, 2.0),    # EI/cost^2 + 0.1*sigma/cost^2
        (0.5, 2.0),    # EI/cost^2 + 0.5*sigma/cost^2
        (0.0, 0.5),    # Pure EI / sqrt(cost)
    ]
    
    for beta, cost_power in param_candidates:
        if _remaining() < 0.05:
            break
        
        # Cost normalization
        cost_norm = np.power(np.maximum(cost, _EPS), cost_power)
        
        # Exploration bonus: sigma / cost
        exploration = sigma / cost_norm
        
        # Combine EI and exploration
        scores = ei_base / cost_norm + beta * exploration
        
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Quality proxy: we want scores to have meaningful variation
        # Use a combination that rewards both spread and magnitude
        s_mean = np.mean(scores)
        s_std = np.std(scores)
        
        if s_mean < 1e-15 and s_std < 1e-15:
            proxy = -1e6
        else:
            # Reward moderate spread and positive mean
            proxy = s_mean * (1.0 + np.log1p(s_std))
        
        if proxy > best_proxy:
            best_proxy = proxy
            best_scores = scores
    
    if best_scores is None:
        # Fallback
        cost_norm = np.maximum(cost, _EPS)
        best_scores = ei_base / cost_norm
        best_scores = np.nan_to_num(best_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_scores
# EVOLVE-BLOCK-END
