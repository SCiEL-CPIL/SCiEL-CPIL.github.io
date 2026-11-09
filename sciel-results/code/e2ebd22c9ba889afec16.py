"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition scores for a given set of candidate points.
    
    This implementation uses the Upper Confidence Bound (UCB) strategy
    with an adaptive parameter beta. For minimization, we maximize the optimistic
    lower bound: Score = -mean + beta * std.
    
    The beta parameter is adapted based on the spread of mu values to balance
    exploration and exploitation. When uncertainty (sigma) is high relative to
    the spread of predictions, we explore more aggressively.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        # Return zeros to ensure finite output and avoid timeout kill mid-calculation
        return np.zeros_like(mu)
    
    # Adaptive UCB parameter for exploration.
    # Use a base beta and scale based on the ratio of sigma to mu spread.
    # This helps adapt the exploration level to the current state of optimization.
    
    # Compute the spread of mu values to understand the current uncertainty landscape
    mu_range = np.ptp(mu)
    sigma_mean = np.mean(sigma)
    
    # If mu_range is very small, we're in a region of low predictive variance
    # If sigma_mean is high relative to mu_range, we should explore more
    if mu_range > 1e-8:
        # Scale beta: higher when sigma is large relative to mu spread
        # This encourages exploration in uncertain regions
        sigma_to_mu_ratio = sigma_mean / (mu_range + 1e-8)
        beta = 1.2 + 0.8 * min(sigma_to_mu_ratio, 2.0)
    else:
        # If mu is nearly constant, use a moderate exploration level
        beta = 1.5
    
    # Calculate UCB scores for minimization
    # We want to find points with potentially low f(x).
    # The optimistic lower bound is mu - beta*sigma.
    # We maximize this "potential" to select the next point.
    # Thus, Score = -(mu - beta*sigma) = -mu + beta*sigma
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
