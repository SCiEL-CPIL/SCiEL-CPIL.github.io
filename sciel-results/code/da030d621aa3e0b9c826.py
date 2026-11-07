"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 320  # seconds, safety margin before 330s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    an exploration bonus proportional to the standard deviation (sigma).
    
    This hybrid approach (EI + alpha * sigma) balances the exploitation of
    promising regions (via EI) with the exploration of uncertain regions
    (via the sigma term), which helps escape local minima in the 6D Hartmann
    function.
    """
    if _time_left() < 5:
        # Fallback: return pure exploit (lowest mu) if time is running out
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Exploration Bonus
    # We add a term proportional to sigma to encourage exploration.
    # A coefficient of 0.5 is a reasonable starting point for balancing
    # exploitation and exploration in this dimensionality and budget.
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma_safe
    
    # 3. Combine scores
    # Higher score means more likely to be evaluated next.
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final safety check
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e10)
        
    return scores
# EVOLVE-BLOCK-END
