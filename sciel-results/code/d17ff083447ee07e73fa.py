# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic UCB strategy.
    
    Instead of EI, we use a Upper Confidence Bound (UCB) approach:
    score = (f_best - mu) + kappa * sigma
    This encourages exploration in uncertain regions.
    
    Key structural difference: The exploration coefficient kappa is not constant.
    It is scaled by the inverse of the cost of the current best point (approximated 
    by the mean cost of low-mu points or simply using a dynamic factor based on 
    the relationship between f_best and the costs).
    
    Actually, a simpler and robust structural change is to use a 
    "Cost-Adjusted UCB" where we balance exploitation (f_best - mu) 
    and exploration (sigma) using a cost-aware trade-off parameter.
    
    Let's use: score = (f_best - mu) / cost + kappa * sigma / cost
    But to make it structurally different from EI/cost, we drop the 
    probabilistic EI calculation and use a heuristic UCB with a 
    cost-normalized exploration term.
    
    Strategy:
    1. Compute the "regret" potential: (f_best - mu).
    2. Compute the "uncertainty": sigma.
    3. Combine them as: score = (f_best - mu) + beta * sigma.
    4. Normalize by cost: score / cost.
    
    To add the "dynamic" aspect: If the current best point was expensive, 
    we should explore more. We can estimate the "cost of best" by looking 
    at the cost of points with low mu. However, we don't have the exact 
    cost of the best point, only the array of costs for candidates.
    
    A better structural change: Use a **Hypervolume-inspired** or 
    **Multi-objective** view. Minimize f, minimize cost.
    But the task asks for a single score.
    
    Let's try a **Bayesian Regret** proxy with a different kernel for cost.
    Current: EI / cost^0.7 + bonus.
    New: UCB / cost.
    
    UCB = (f_best - mu) + kappa * sigma.
    Score = UCB / cost.
    
    We will tune kappa dynamically. A high kappa is good for early stages 
    (high uncertainty) or when the best point is expensive.
    Since we don't know the stage or best cost directly, we can use 
    the ratio of the max possible improvement to the current f_best 
    or simply a fixed, well-tuned kappa for UCB which behaves differently 
    than EI (EI is bounded and saturates, UCB is linear in sigma).
    
    Let's implement: Score = ((f_best - mu) + kappa * sigma) / cost
    with kappa = 1.0 (in objective units).
    
    To ensure stability and handle the "cost" aspect more structurally, 
    we can also add a small epsilon to cost to avoid division by zero, 
    and clip sigma to prevent extreme values.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB term: Exploitation + Exploration
    # f_best - mu is the expected improvement if we assume the mean is the truth.
    # kappa * sigma is the exploration bonus.
    # Using kappa=1.0 is a standard choice for UCB in BO.
    kappa = 1.0
    
    ucb = (f_best - mu) + kappa * sigma_safe
    
    # Cost normalization
    # Dividing by cost favors cheap points.
    # To make it structurally different from the current EI/cost^0.7, 
    # we use a linear cost penalty in the denominator, but we also 
    # add a small "floor" to the score to prevent negative scores from 
    # dominating if mu > f_best (which can happen with UCB if sigma is large 
    # and mu is high, but f_best - mu is negative).
    # Actually, if mu > f_best, the exploitation term is negative.
    # EI handles this by being 0 or negative. UCB can be negative.
    # We want higher scores to be better.
    # If a point is predicted to be worse than f_best (mu > f_best), 
    # its exploitation term is negative.
    # However, if sigma is large, the exploration term might make it positive.
    
    scores = ucb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
