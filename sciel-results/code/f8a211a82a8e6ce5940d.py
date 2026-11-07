"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we stay within limits
_start_time = time.time()
_TIMEOUT_MARGIN = 5.0  # seconds to spare before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural approach:
    Combining Expected Improvement with a cost-normalized exploration bonus
    that explicitly rewards high-uncertainty regions, using a different
    formulation than the simple EI/LCB hybrid.
    
    Instead of averaging EI and LCB, we use:
    - A "cost-adjusted expected improvement" that accounts for the fact
      that high-cost regions are penalized more heavily
    - An explicit exploration term based on sigma that is NOT linear in sigma
      but uses a saturating function to avoid over-exploring
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
        
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    # Check time budget - if we're close to timeout, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # 330 - 5 second margin
        return np.zeros_like(mu)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: cost is 1 + mean(x_scaled), so it's in [1, 2]
    # We want to penalize high cost, but not too aggressively
    cost_safe = np.maximum(cost, 1e-10)
    
    # Structural change: Use a "cost-weighted" EI with an exploration bonus
    # that is based on the relative uncertainty compared to the scale of f_best
    
    # Scale of the objective: use |f_best| as a reference, with a floor
    f_scale = max(abs(f_best), 1e-6)
    
    # Exploration term: sigma relative to the objective scale, with a cap
    # This creates a saturating exploration bonus
    sigma_rel = sigma_safe / f_scale
    # Use a soft cap: tanh(sigma_rel) to prevent unbounded exploration
    exploration = np.tanh(sigma_rel)
    
    # Combine: cost-adjusted EI + weighted exploration
    # The EI term is divided by cost (cost-aware exploitation)
    # The exploration term is also divided by cost (cost-aware exploration)
    # But we weight exploration more heavily in early stages
    
    # Determine exploration weight based on how good f_best is
    # If f_best is close to 0 (good), we can be more exploitative
    # If f_best is far from 0, we need more exploration
    # f_best is the minimum of three-hump-camel, which has a global min near 0
    # and local minima around -1.86. So f_best in [-1.86, 0] is typical.
    
    # Use a simple heuristic: exploration weight decreases as f_best improves
    # When f_best is close to 0, weight ~0.3; when f_best is -1.86, weight ~0.7
    if f_best > 0:
        exp_weight = 0.7
    else:
        # Linear interpolation from 0.7 at f_best=-1.86 to 0.3 at f_best=0
        exp_weight = 0.3 + 0.4 * min(1.0, max(0.0, (-f_best) / 1.86))
    
    # Final score
    ei_term = ei / cost_safe
    exp_term = exploration / cost_safe
    
    scores = (1.0 - exp_weight) * ei_term + exp_weight * exp_term
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
