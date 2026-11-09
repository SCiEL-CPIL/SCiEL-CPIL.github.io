"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_ACQ_START_TIME = None
_DEADLINE = 325.0  # seconds, safety margin before 330s kill
_SAFETY_MARGIN = 5.0


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        return False
    elapsed = time.time() - _ACQ_START_TIME
    return elapsed > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement with adaptive exploration based on posterior structure.
    """
    global _ACQ_START_TIME
    _ACQ_START_TIME = time.time()
    
    n = len(mu)
    
    # Check deadline early
    if _check_deadline():
        # Fallback: pure exploitation
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Standard EI
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Adaptive exploration: use UCB with a kappa that adapts to the posterior
    # When sigma is high relative to the range of mu, we're in an uncertain region
    # and should explore more. When sigma is low, exploit.
    
    # Compute a normalized uncertainty measure
    sigma_max = sigma.max() if sigma.max() > 0 else 1.0
    mu_range = mu.max() - mu.min() if mu.max() > mu.min() else 1.0
    
    # Normalize sigma to [0, 1]
    sigma_norm = sigma / (sigma_max + 1e-10)
    
    # Adaptive kappa: higher when uncertainty is high
    # This encourages exploration in uncertain regions
    kappa = 1.0 + 2.0 * sigma_norm
    
    # UCB component
    ucb = mu + kappa * sigma
    
    # Combine EI and UCB: weight EI more when we have good predictions, UCB when uncertain
    # Use a soft blend based on how confident we are
    confidence = 1.0 / (1.0 + sigma_norm)  # higher when sigma is low
    
    # Final score: blend of EI (exploitation) and negative UCB (exploration)
    # Note: UCB is for maximization, but we're minimizing, so we use -UCB
    # Actually, let's think about this differently.
    # EI is already a good acquisition for minimization.
    # Let's add a diversity bonus based on how unique each point's posterior is.
    
    # Diversity bonus: points with unique (mu, sigma) pairs get a bonus
    # Use a simple approach: normalize mu and sigma, then compute distance to centroid
    mu_center = np.mean(mu)
    sigma_center = np.mean(sigma)
    
    mu_norm = (mu - mu_center) / (mu_range + 1e-10)
    sigma_norm2 = (sigma - sigma_center) / (sigma_max + 1e-10)
    
    # Distance from centroid in normalized space
    dist_from_centroid = np.sqrt(mu_norm**2 + sigma_norm2**2)
    
    # Normalize distance to [0, 1]
    dist_max = dist_from_centroid.max() if dist_from_centroid.max() > 0 else 1.0
    dist_norm = dist_from_centroid / dist_max
    
    # Diversity bonus: add a small bonus for points far from the centroid
    diversity_bonus = 0.2 * dist_norm * sigma_max
    
    # Final score: EI + diversity bonus
    scores = ei + diversity_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline before returning
    if _check_deadline():
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
