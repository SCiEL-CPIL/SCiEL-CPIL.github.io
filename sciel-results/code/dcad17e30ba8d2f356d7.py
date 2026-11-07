import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Distance-weighted Expected Improvement.
    
    This approach combines Expected Improvement with a distance-based exploration bonus.
    The distance is measured in the standardized input space (mu and sigma are already
    computed at candidate points), and we use the spread of sigma as a proxy for
    unexplored regions. Points far from high-density areas of exploration get boosted.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle sigma near zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For minimization, EI is already positive when mu < f_best
    # We want to maximize EI
    
    # Now add a distance-based exploration bonus
    # Use sigma as a proxy: high sigma means unexplored/uncertain regions
    # We create a bonus that rewards points with high uncertainty
    
    # Normalize sigma to [0, 1] range for scaling
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus: weighted by normalized sigma
    # This creates a repulsive force that pushes exploration to uncertain regions
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma_norm * np.max(EI)
    
    # Combine EI and exploration bonus
    score = EI + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
