"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _safe_z(z):
    """Clamp z to avoid numerical issues in norm.cdf."""
    return np.clip(z, -10, 10)

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 325:  # 5s safety margin
        # Fallback: simple cost-aware exploitation
        return -mu / (cost + 1e-9)
    
    # Structural Change: Use Probability of Improvement (PI) instead of Expected Improvement (EI).
    # PI is more robust and less sensitive to scale issues in the tails.
    # We combine PI with an explicit exploration bonus to handle multimodality.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    z = _safe_z((f_best - mu) / sigma_safe)
    
    # Probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # Cost-aware PI: prefer points with high probability of improvement and low cost
    score_pi = pi / (cost + 1e-9)
    
    # Exploration Bonus: 
    # Encourage exploration of high-uncertainty regions.
    # This helps escape local minima in the Griewank function.
    # The term is scaled to be comparable to the PI term (which is ~0 to 1).
    # We use a small coefficient to ensure PI dominates, but exploration still influences selection.
    explore_bonus = 0.15 * sigma_safe / (cost + 1e-9)
    
    # Combine scores
    score = score_pi + explore_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
