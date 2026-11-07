# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) for minimization.
    We want to maximize the score.
    For minimization, we use: Score = -mu + beta * sigma
    This encourages picking points with low mean (exploitation) 
    and high uncertainty (exploration).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB coefficient
    # A value around 2-3 is standard. 
    # Given the short length scale and small domain, we might need 
    # slightly higher exploration initially, but UCB adapts via sigma.
    beta = 2.5
    
    # Calculate UCB score for minimization
    # We want to find the minimum, so we "maximize" the potential to be low.
    # The "upper bound" of the regret is roughly related to -mu + beta*sigma.
    # Alternatively, think of it as: we want to minimize f(x).
    # We estimate f(x) ~ N(mu, sigma).
    # We want to select x that maximizes the probability of being better than f_best?
    # That's PI.
    # UCB is a heuristic that works well.
    # Score = -mu + beta * sigma.
    # If mu is 0 and sigma is 1, Score = 2.5.
    # If mu is 10 and sigma is 1, Score = -10 + 2.5 = -7.5.
    # We pick the highest score. So we pick mu=0. Correct.
    # If mu is 5 and sigma is 10, Score = -5 + 25 = 20.
    # If mu is 0 and sigma is 1, Score = 2.5.
    # We pick the uncertain point (mu=5, sigma=10) because it might be lower.
    
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
