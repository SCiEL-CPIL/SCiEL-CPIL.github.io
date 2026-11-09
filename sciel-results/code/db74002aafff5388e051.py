"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
DEADLINE_MARGIN = 10  # seconds

def acquisition(mu, sigma, f_best):
    """
    Pure Posterior Mean Minimization (PMM) strategy.
    
    This approach completely ignores the predictive standard deviation (sigma) 
    and focuses solely on minimizing the predictive mean (mu). 
    By returning -mu, the evaluator will select the candidate with the lowest 
    predicted value. This is a deterministic exploitation strategy that 
    differs from probabilistic sampling (TS) or uncertainty-weighted methods 
    (EI/UCB) by not balancing exploration via uncertainty.
    
    For the sphere function f(x) = x^2 on [-5, 5], the global minimum is at x=0.
    PMM tends to converge quickly to the region of the global minimum once the 
    GP has sufficient information, avoiding the conservative bias of EI which 
    might keep exploring areas with high uncertainty but high mean values.
    """
    # Check deadline to ensure we finish within the time limit
    # The evaluator kills after 330s. We add a safety margin.
    if time.time() - _start_time > 330 - DEADLINE_MARGIN:
        # If we are running out of time, return a simple fallback immediately
        # to avoid any potential overhead in the return path, though -mu is already simple.
        return -mu

    # Ensure mu is finite. If not, replace with a large penalty value.
    # However, the contract states mu and sigma are computed by the evaluator,
    # so they should generally be finite. We add a safeguard just in case.
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Return -mu so that higher scores correspond to lower predicted values (minimization).
    score = -mu_clean
    
    # Final check for finiteness of the score itself
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
