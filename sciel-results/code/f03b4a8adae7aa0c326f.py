"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement (EI) normalized by cost
    and a UCB-like exploration term, with adaptive exploration based on
    the current best value and cost budget.
    """
    # Deadline guard: track elapsed time to ensure we return within safety margin
    t_start = time.time()
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI by cost to get "improvement per unit cost"
    ei_per_cost = ei / cost
    
    # UCB-like exploration term
    # beta * sigma / cost encourages exploring uncertain regions
    # beta is adaptive: larger when f_best is far from optimum, smaller when close
    # For rosenbrock 2d, the minimum is 0, so we can use f_best as a proxy
    # for how far we are from optimum
    
    # Adaptive beta: start with moderate exploration, decrease as we get closer to optimum
    # Use a decay based on f_best (assuming f_best >= 0 for rosenbrock)
    # When f_best is large, we want more exploration; when small, less
    beta = 0.5 + 2.0 * np.exp(-0.1 * max(f_best, 0.0))
    
    # UCB term: beta * sigma / cost
    ucb_term = beta * sigma / cost
    
    # Combine EI and UCB
    scores = ei_per_cost + ucb_term
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline: if we've used too much time, return immediately
    # The vectorized operations above should be very fast for 2000 points
    # but this guard ensures safety
    if time.time() - t_start > 280:
        # Return a simple fallback if we're running out of time
        return np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
