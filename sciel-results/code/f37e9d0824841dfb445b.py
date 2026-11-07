import numpy as np
import time
from scipy.stats import norm

_start_time = None
_DEADLINE = 325.0  # 5 seconds margin from 330s

def acquisition(mu, sigma, f_best):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard
    if elapsed > _DEADLINE:
        # Return a simple, fast score. Pure exploitation is a safe fallback.
        return -mu
    
    # Knowledge Gradient (KG) approximation for minimization
    # KG estimates the expected reduction in the best value after evaluating at a point.
    # For a single candidate, KG ≈ E[max(0, f_best - f(x))] but with better exploration
    # properties than EI because it considers the value of information.
    
    # A practical KG approximation:
    # 1. Compute the probability that f(x) < f_best: P = Phi((f_best - mu) / sigma)
    # 2. The expected improvement if we evaluate here
    # 3. KG ≈ EI * (1 + exploration_bonus)
    #    where exploration_bonus accounts for uncertainty in the current best
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_base - mu) / sigma_safe if False else (f_best - mu) / sigma_safe
    
    # Standard EI component
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # KG adjustment: add a term that rewards high uncertainty when current best is uncertain
    # The idea: if sigma is large, there's more to learn, so we should explore more
    # This is a heuristic approximation of the true KG
    
    # Compute a proxy for the value of information
    # When sigma is large relative to the potential improvement, exploration is more valuable
    improvement = np.maximum(f_best - mu, 0)
    
    # Exploration bonus: scales with sigma but is bounded
    # Use a smooth function that increases with sigma but saturates
    exploration_bonus = sigma_safe * norm.pdf(z) * 0.5
    
    # Combine: KG ≈ EI + exploration_bonus
    kg = ei + exploration_bonus
    
    # Ensure finite scores
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
