"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    Instead of a fixed linear combination of EI and UCB, use a 
    Thompson sampling-inspired approach that adaptively balances
    exploitation and exploration based on the current state.
    
    The key structural change: use a probabilistic approach where
    the exploration weight adapts to the ratio of f_best to the
    range of mu values, reflecting how much improvement is still possible.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Return a safe fallback: just use cost-normalized EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * np.maximum(0, z)  # approximate CDF for speed
        ei_cost = ei / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return score
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard EI computation
    # Use stable computation for large |z|
    ei = np.zeros_like(z)
    mask = z > 0
    # For positive z: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei[mask] = (f_best - mu[mask]) * _norm_cdf(z[mask]) + sigma[mask] * _norm_pdf(z[mask])
    # For negative z: EI is small but positive, use direct formula
    ei[~mask] = (f_best - mu[~mask]) * _norm_cdf(z[~mask]) + sigma[~mask] * _norm_pdf(z[~mask])
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Adaptive exploration: use a sigmoid-based weight that increases
    # exploration when f_best is still far from the expected minimum
    # (i.e., when mu values are still high relative to f_best)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # If f_best is close to mu_min, we're exploiting; if far, explore more
    # Normalize f_best relative to the mu range
    f_best_norm = (f_best - mu_min) / (mu_range + 1e-9)
    
    # Exploration weight: high when f_best_norm is high (still lots of room)
    # Use a smooth transition centered around 0.5
    exploration_weight = 1.0 / (1.0 + np.exp(-4.0 * (f_best_norm - 0.5)))
    
    # UCB term
    beta = 2.5
    ucb_cost = (beta * sigma) / cost_safe
    
    # Combine with adaptive weighting
    score = (1.0 - exploration_weight) * ei_cost + exploration_weight * ucb_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score


def _norm_cdf(x):
    """Fast approximation of standard normal CDF."""
    # Use the error function approximation
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast computation of standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x):
    """Approximation of the error function using Abramowitz and Stegun."""
    # Coefficients
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = np.sign(x)
    x_abs = np.abs(x)
    
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    
    return sign * y
# EVOLVE-BLOCK-END
