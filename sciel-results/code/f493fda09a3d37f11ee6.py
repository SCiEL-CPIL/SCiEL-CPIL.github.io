"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time and step count
_start_time = None
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _step_count
    
    # Initialize global state on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    # Deadline guard: The evaluator kills after 330s. 
    # We have 25 evaluations (30 total - 5 initial). 
    # Each call should be fast, but we add a safety margin.
    # If we are close to the deadline, we could potentially degrade to a faster method,
    # but UCB is already O(N) and very fast. 
    # However, if we detect a hang or excessive time, we return a safe fallback.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety fallback: return uniform random-like scores to avoid stalling
        # In reality, the function returns before this if the loop is efficient.
        # This is a hard stop to prevent timeout if something unexpected happens.
        return np.ones_like(mu)
    
    _step_count += 1
    
    # Total evaluations allowed: 30. Initial: 5. Loop iterations: 25.
    # step_count here corresponds to the loop iteration index (1-based for the first loop call)
    # Let's define progress as step_count / 25
    progress = _step_count / 25.0
    
    # Adaptive beta for UCB
    # Early on (low progress), we want more exploration (higher beta).
    # Late on (high progress), we want more exploitation (lower beta).
    # A common schedule is beta ~ 1/sqrt(t) or exponential decay.
    # Let's use a decay that starts high and drops significantly.
    # At step 1 (progress ~ 0.04), beta ~ 2.0
    # At step 25 (progress ~ 1.0), beta ~ 0.5
    
    # Linear interpolation in log-space for beta
    log_beta_start = np.log(2.0)
    log_beta_end = np.log(0.5)
    log_beta = log_beta_start + (log_beta_end - log_beta_start) * progress
    beta = np.exp(log_beta)
    
    # Calculate UCB
    # Standardize sigma to avoid scale issues if the GP is not perfectly normalized
    # However, the contract says "standardized targets", so sigma is in objective units.
    # Ackley function range is roughly [0, ~28]. 
    # If sigma is very small, UCB behaves like EI.
    # If sigma is large, UCB favors high uncertainty.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB = mu + beta * sigma
    # Note: We are minimizing. So we want low mu.
    # Standard UCB for minimization: score = mu - beta * sigma ?
    # No, UCB is for maximizing a reward. Here we minimize cost.
    # Let's convert to a "regret" or "improvement" perspective or just flip signs.
    # If we define "score" such that higher is better for the acquisition:
    # We want to select points with LOW mu and HIGH sigma.
    # So, score = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
