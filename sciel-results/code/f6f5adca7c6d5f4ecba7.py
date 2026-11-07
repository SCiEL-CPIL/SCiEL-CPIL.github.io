"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    delta = f_best - mu
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    
    # UCB exploration term: beta * sigma
    # Use adaptive beta based on how much we've explored
    # Early: more exploration, late: more exploitation
    # Use the spread of mu to gauge convergence
    mu_std = np.std(mu)
    mu_range = np.ptp(mu)
    
    # Adaptive beta: larger when uncertainty is high relative to improvement scale
    # Normalize sigma by typical scale of the problem
    sigma_scale = np.maximum(mu_std, 1e-6)
    normalized_sigma = sigma_safe / sigma_scale
    
    # Beta that balances exploration: decays as f_best improves
    # Use a fixed beta that works well across all four functions
    beta = 0.5 + 0.5 * np.exp(-np.abs(delta) / (sigma_scale + 1e-6))
    
    # UCB score
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware: normalize both EI and UCB by cost
    ei_score = ei / cost_safe
    ucb_score = ucb / cost_safe
    
    # Blend: weight EI more when we have good estimates, UCB when uncertain
    # Use the ratio of EI to UCB as a confidence measure
    # When EI >> UCB, trust EI; when they're close, blend
    # Simple approach: weighted average with cost adjustment
    # Higher sigma relative to delta means we're less certain -> more UCB
    
    # Confidence: how much improvement we expect vs. uncertainty
    confidence = np.abs(delta) / (sigma_safe + 1e-6)
    # Clip confidence to [0, 1]
    confidence = np.clip(confidence, 0, 1)
    
    # Weight: more EI when confident, more UCB when uncertain
    w_ei = 0.3 + 0.7 * confidence  # 0.3 to 1.0
    w_ucb = 1.0 - w_ei            # 0.7 to 0.0
    
    # Combined score
    score = w_ei * ei_score + w_ucb * ucb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
