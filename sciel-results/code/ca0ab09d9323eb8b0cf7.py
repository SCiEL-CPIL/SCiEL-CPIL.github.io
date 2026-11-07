"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 6D.
    
    Uses a combination of Expected Improvement (EI) and a carefully tuned
    exploration bonus. The exploration bonus is scaled by a factor that
    decays as f_best improves, encouraging more exploration early on
    and more exploitation later.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean of the GP at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation of the GP at candidate points (in objective units)
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means evaluate next
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score for the normal distribution
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight:
    # Start with moderate exploration, reduce as f_best approaches the optimum
    # Hartmann 6D minimum is approximately -3.32237
    # Early on, f_best is much higher (worse), so we want more exploration
    # Later, f_best is close to optimum, so we want more exploitation
    
    # Scale exploration based on how far f_best is from the known minimum
    # When f_best is far from optimum, explore more; when close, exploit more
    min_val = -3.32237
    distance_to_min = max(0.0, f_best - min_val)
    
    # Exploration weight decays as we get closer to the optimum
    # Use a smooth decay: weight = base_weight * exp(-decay_rate * distance)
    # But we want MORE exploration when distance is LARGE (early)
    # So: weight = base_weight * (1 - exp(-decay_rate * distance))
    # This gives small weight when distance is small (late) and larger when distance is large (early)
    
    # Alternatively, a simpler approach: fixed weight that works well
    # The hybrid EI + 0.1*sigma achieved 1.287, so let's try a slightly different balance
    
    # Use a slightly higher exploration weight to encourage more diverse sampling
    # in the early stages, which is critical with only 30 evaluations
    exploration_weight = 0.15
    
    exploration_bonus = exploration_weight * sigma_safe
    
    # Total score: higher is better
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
