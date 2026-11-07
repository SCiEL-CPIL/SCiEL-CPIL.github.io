"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    Instead of deterministic Expected Improvement (EI), we use Thompson Sampling.
    We sample a potential objective value from the posterior distribution 
    N(mu, sigma) for each candidate. Since the goal is minimization, we 
    convert the objective to a "score" where higher is better (by negating).
    We then divide by the cost to prioritize cheaper evaluations that are 
    likely to be good.
    
    This approach naturally balances exploration (via sampling from sigma) 
    and exploitation (via mu) while being cost-aware.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety: avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: Draw a sample from the posterior distribution
    # mu and sigma are in objective units (lower is better)
    # We want to maximize the score, so we negate the objective value
    # y_sample ~ N(mu, sigma)
    # score = -y_sample / cost
    
    # Generate random normal noise
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior
    y_sample = mu + sigma * z
    
    # Convert to maximization score and penalize by cost
    # Using division by cost is a common heuristic for cost-aware BO
    scores = -y_sample / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
