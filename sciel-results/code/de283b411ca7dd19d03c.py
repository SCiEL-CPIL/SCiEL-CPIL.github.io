import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Levy 2D.
    
    Strategy: Use UCB with cost normalization. The parent (score 0.2431)
    used UCB with beta=4.0 divided by cost. We refine this by:
    1. Using a slightly lower beta (3.0) to reduce over-exploration
    2. Adding a small EI component for exploitation when GP is confident
    3. Keeping the 1/cost normalization (not 1/cost^2 which was too aggressive)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # UCB for minimization: -mu + beta * sigma
    beta = 3.0
    ucb = -mu + beta * sigma
    
    # Add a small EI component for exploitation
    diff = f_best - mu
    z = diff / sigma
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = diff * (1.0 - cdf) + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # Combine UCB and EI with weighting
    # UCB dominates for exploration, EI adds exploitation signal
    scores = (ucb + 0.5 * ei) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
