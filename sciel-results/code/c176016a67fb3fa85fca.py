import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    Uses a dynamically increasing exploration constant to balance exploration
    and exploitation over the course of the optimization.
    
    UCB = mu - kappa * sigma
    where kappa = c * sqrt(log(n+1))
    
    We estimate n (number of evaluations) from the length of mu/sigma arrays
    and the structure of the problem. Since we don't have direct access to n,
    we use a heuristic based on the spread of sigma values to estimate how
    far along in the optimization we are.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Estimate the number of evaluations so far.
    # In the early stages, sigma values tend to be larger (more uncertainty).
    # As more points are evaluated, sigma tends to decrease in explored regions.
    # We use the mean sigma as a proxy for exploration level.
    
    # A simpler approach: use a fixed schedule based on the problem structure.
    # Since we know there are 30 total evaluations and 5 initial points,
    # we can estimate the current step from the pattern of sigma values.
    
    # For a robust approach, let's use a dynamic kappa that starts high
    # and decreases as we gain confidence.
    
    # The mean sigma gives us a sense of overall uncertainty.
    # Early on, mean sigma is high; later, it's lower.
    # We can normalize by the initial expected sigma.
    
    # For Hartmann 3D with Matern-5/2 and length scale 0.2,
    # the initial uncertainty (prior std) is around 1.0 (standardized targets).
    
    mean_sigma = np.mean(sigma_safe)
    
    # Estimate progress: if mean_sigma is close to 1, we're early;
    # if it's much smaller, we're later.
    # kappa should start around 2-3 and decrease to around 0.5-1.
    
    # Use a smooth interpolation:
    # kappa = base + amplitude * exp(-decay * (1 - mean_sigma))
    # When mean_sigma ~ 1 (early), kappa ~ base + amplitude
    # When mean_sigma ~ 0 (late), kappa ~ base
    
    base_kappa = 0.5
    amplitude = 2.5
    decay = 3.0
    
    # mean_sigma ranges roughly from ~1 (early) to ~0.1-0.3 (late)
    # (1 - mean_sigma) ranges from ~0 to ~0.7-0.9
    
    kappa = base_kappa + amplitude * np.exp(-decay * (1.0 - mean_sigma))
    
    # Calculate UCB scores: mu - kappa * sigma
    # Lower is better for the objective, so we want to maximize (mu - kappa*sigma)
    # i.e., find points where the mean is low AND uncertainty is high.
    
    scores = mu - kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
