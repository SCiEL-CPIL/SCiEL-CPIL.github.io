import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Combines the structural insight of using a linear cost division (LCB-style
    robustness to GP errors) with the refinement machinery of a stochastic
    cost model. The core acquisition is EI / cost (linear division, not power),
    which allows high-improvement high-cost points to be considered if the
    improvement is sufficient. A small exploration bonus weighted by 1/cost
    encourages exploring cheap, uncertain regions.
    
    The linear cost division (vs power) is the key structural change: it is
    more robust to GP prediction errors and scaling, and does not suppress
    high-improvement points as aggressively as cost^alpha with alpha > 1.
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        return 1.0 / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement: positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Linear cost division: EI / cost
    # This is the key structural choice - linear division rather than power
    # allows high-improvement high-cost points to be considered
    score = ei / cost
    
    # Add a small exploration bonus: prefer high-sigma regions when EI is low
    # This helps in early stages
    # Use a relative sigma: sigma / (|f_best| + 1) to make it scale-invariant
    sigma_scale = np.abs(f_best) + 1.0
    rel_sigma = sigma_safe / sigma_scale
    
    # Exploration bonus: small, proportional to relative sigma, weighted by 1/cost
    # This encourages exploring cheap, uncertain regions
    exploration_bonus = 0.05 * rel_sigma / cost
    
    score = score + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
