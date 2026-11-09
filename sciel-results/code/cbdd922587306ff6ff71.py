"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for adaptive beta
_STEP_COUNT = 0
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition for 1D sphere minimization.
    
    Uses a UCB-style score: score = -mu + beta * sigma
    where beta adapts based on the number of acquisition calls (iterations).
    
    - Early iterations: Higher beta for exploration.
    - Later iterations: Lower beta for exploitation.
    
    This directly optimizes for low predicted values while allowing
    uncertainty-driven exploration early on.
    """
    global _STEP_COUNT
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple UCB with fixed low beta
        sigma_safe = np.maximum(sigma, 1e-12)
        score = -mu + 0.05 * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Increment step counter
    _STEP_COUNT += 1
    
    # Numerical safety for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: exponential decay
    # We expect ~25 optimization steps (30 total - 5 initial).
    # Start with beta=0.5, decay to beta=0.05
    # alpha=0.15 gives half-life of ~4.6 steps
    beta_max = 0.5
    beta_min = 0.05
    alpha = 0.15
    
    # If step count exceeds expected, keep beta_min
    t = min(_STEP_COUNT, 25)
    beta = beta_min + (beta_max - beta_min) * np.exp(-alpha * t)
    
    # UCB score for minimization: higher score = more likely to be good
    # We want low mu and high sigma (uncertainty might hide a better value)
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
