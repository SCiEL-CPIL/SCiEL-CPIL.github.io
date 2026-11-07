# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Use Probability of Improvement (PoI) as base
    # PoI is more robust than EI because it's bounded in [0,1] and
    # focuses on the likelihood of finding a better point rather than expected gain
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)  # P(f(x) < f_best)
    
    # Adaptive exploration: use sigma-rank bonus
    # Rank sigmas from 0 (lowest) to 1 (highest)
    sigma_rank = np.argsort(np.argsort(sigma_safe)).astype(float) / (n - 1) if n > 1 else np.zeros(n)
    
    # Adaptive beta based on convergence signal
    # frac_better: fraction of points with predicted mean below f_best
    frac_better = np.mean(mu < f_best)
    
    # When we're confident (high frac_better), exploit more (lower beta)
    # When uncertain (low frac_better), explore more (higher beta)
    beta = 0.5 + 1.5 * (1.0 - frac_better)
    beta = np.clip(beta, 0.3, 2.0)
    
    # UCB component for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine PoI and UCB with sigma-rank exploration bonus
    # Normalize UCB to [0,1]
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros(n)
    
    # PoI is already in [0,1]
    # Weight: start with more PoI (exploitation), add exploration via sigma_rank
    # The sigma_rank bonus encourages visiting high-uncertainty regions
    # Weight PoI heavily when we have good predictions, less when uncertain
    w_poi = 0.6 + 0.3 * frac_better
    w_ucb = 0.4 - 0.3 * frac_better
    w_rank = 0.15  # Fixed exploration bonus weight
    
    score = w_poi * poi + w_ucb * ucb_norm + w_rank * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
