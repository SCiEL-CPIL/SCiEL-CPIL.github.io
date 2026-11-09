"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

# Track call count to adapt beta
_call_count = [0]
_TOTAL_EVALS = 30  # 5 initial + 25 acquisition-driven
_INITIAL_EVALS = 5

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses GP-LCB (Lower Confidence Bound) with an adaptive exploration
    parameter that decays from high (early exploration) to low (late
    exploitation) based on the fraction of the budget consumed.
    
    For minimization: score = -(mu - beta * sigma) = -mu + beta * sigma
    Higher score means more likely to be the minimum.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    _call_count[0] += 1
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: decays from beta_max to beta_min over the evaluation budget
    # beta_max: high exploration early on
    # beta_min: low exploration, near-pure exploitation late
    beta_max = 4.0
    beta_min = 0.5
    
    # Fraction of budget consumed (including initial evals)
    evals_used = _call_count[0] + _INITIAL_EVALS
    frac = min(evals_used / _TOTAL_EVALS, 1.0)
    
    # Exponential decay of beta: beta = beta_min + (beta_max - beta_min) * exp(-k * frac)
    # k controls the speed of decay; k=3 gives beta ~ beta_min + 0.05*(beta_max-beta_min) at frac=1
    k = 3.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-k * frac)
    
    # GP-LCB for minimization:
    # We want to find points likely to have low values.
    # Lower bound on f(x): mu(x) - beta * sigma(x)
    # We want to MINIMIZE this lower bound, so we maximize its negation.
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
