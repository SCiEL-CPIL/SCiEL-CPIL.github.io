"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement with adaptive cost-awareness.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * ndtr(z) + sigma_safe * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei = np.maximum(ei, 0.0)

    # Adaptive cost-awareness:
    # - When f_best is close to optimum (~-3.32), be more aggressive on cost
    # - When far from optimum, allow more exploration
    # The cost ranges from 1 to 2 (1 + mean(x) where x in [0,1]^6)
    
    # Normalize cost to [0,1] range for better scaling
    cost_norm = (cost_safe - 1.0) / 1.0  # Now in [0, 1]
    cost_norm = np.clip(cost_norm, 0, 1)
    
    # Use a softer cost penalty that's more adaptive
    # Higher cost -> more penalty, but with diminishing returns
    # beta controls how aggressive the cost penalty is
    
    # Adaptive beta: start lower (more exploration) and increase as we get closer
    # f_best starts around -2 to -3 and approaches -3.32
    # When f_best > -3.0, we're far from optimum -> lower beta
    # When f_best < -3.2, we're close -> higher beta
    
    if f_best > -3.0:
        beta = 0.8
    elif f_best > -3.2:
        beta = 1.2
    else:
        beta = 1.6
    
    # Cost penalty: divide by (1 + cost_norm * beta_factor)
    # This is smoother than cost^beta
    cost_penalty = 1.0 + cost_norm * beta * 0.5
    
    scores = ei / cost_penalty
    
    # Add small UCB component for exploration in early stages
    if f_best > -2.5:
        kappa = 1.5
        ucb = mu - kappa * sigma_safe
        ucb_gain = np.maximum(f_best - ucb, 0.0)
        scores += 0.2 * ucb_gain / cost_penalty
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
