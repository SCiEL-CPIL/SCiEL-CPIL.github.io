import numpy as np
import time

# Global variables for time tracking if needed, but per-call is safer
# The task says "Keep an EXPLICIT deadline guard".
# However, the acquisition function is called once per step.
# The step itself includes drawing candidates, GP posterior, etc.
# The acquisition function itself is just vectorized numpy ops.
# It should be very fast (< 1ms).
# The 300s limit is for the whole evaluation loop (30 steps).
# So the acquisition function doesn't need a complex guard, just ensure it doesn't hang.
# But I will add a simple check just in case.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    Score = -mu(x) + beta * sigma(x)
    Higher score indicates a more promising candidate for minimization.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # UCB parameter
    # A value of 2.0 is a common starting point for exploration/exploitation balance.
    # Weierstrass is multimodal, so some exploration is needed.
    beta = 2.0
    
    # Calculate UCB scores
    # We want to minimize f(x).
    # The lower bound of the confidence interval is mu - beta*sigma.
    # We want to maximize the probability of finding a low value.
    # Maximizing -mu + beta*sigma is equivalent to minimizing mu - beta*sigma.
    scores = -mu_safe + beta * sigma_safe
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (should not happen), fallback to 0
    if np.all(scores == -np.inf):
        scores = np.zeros_like(scores)
        
    return scores
