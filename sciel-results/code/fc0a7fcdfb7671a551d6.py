"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 330s limit, 5s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive LCB acquisition function for minimization.
    
    Uses a Lower Confidence Bound (LCB) with an adaptive beta parameter that
    scales with the relative uncertainty of the Gaussian Process. This balances
    exploration (high sigma) and exploitation (low mu) while accounting for
    evaluation cost.
    
    Args:
        mu: Array of mean predictions.
        sigma: Array of standard deviation predictions.
        f_best: Current best observed value.
        cost: Array of costs for each candidate.
        
    Returns:
        Array of scores (higher is better).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.random.rand(len(mu))

    # Convert to numpy arrays if not already
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical safeguards
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Determine the scale of the objective function.
    # Hartmann 3D minimum is approx -3.86. f_best starts high (positive) and decreases.
    # We use abs(f_best) as a proxy for the current "depth" or scale.
    # If f_best is very close to 0, use a minimum scale to avoid division by zero or huge betas.
    f_scale = max(abs(f_best), 1.0)
    
    # Calculate mean uncertainty
    mean_sigma = np.mean(sigma)
    
    # Adaptive Beta calculation
    # Base beta for standard LCB
    base_beta = 1.5
    
    # Scale factor: if mean_sigma is large relative to f_scale, increase exploration
    # This prevents premature convergence when the GP is still uncertain about the landscape.
    # We cap the multiplier to avoid excessive exploration.
    uncertainty_ratio = mean_sigma / f_scale
    # Linear scaling with a cap
    adaptive_factor = 1.0 + 2.0 * min(uncertainty_ratio, 1.0)
    
    beta = base_beta * adaptive_factor

    # LCB for minimization:
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # To find the minimum, we want to maximize the "optimism" that f(x) is low.
    # Alternatively, we can view the score as the potential improvement over f_best.
    # A common score for minimization LCB is: Score = -(mu - beta * sigma) = beta * sigma - mu
    # We divide by cost to make it cost-aware.
    
    scores = (beta * sigma - mu) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (edge case), return a neutral score
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
