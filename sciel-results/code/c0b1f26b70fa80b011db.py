# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    
    STRUCTURAL CHANGE: Uses a Ratio-based acquisition function (EI + Exploration) / Cost.
    Score(x) = (EI(x) + beta * sigma(x)) / (cost(x) + epsilon)
    
    This model prioritizes points that offer high expected improvement relative to their cost.
    Unlike additive penalties, this ratio naturally scales the exploration and improvement
    terms by the cost, ensuring that expensive points must offer significantly higher
    expected improvement to be selected. This aligns with the 'improvement per unit cost'
    objective.
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Small epsilon to avoid division by zero issues, though cost >= 1
    eps = 1e-6

    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # 2. Compute Exploration Term
    # We add a scaled sigma to the numerator. This encourages exploring uncertain regions.
    # The scale of sigma relative to EI determines the exploration-exploitation balance.
    # We use a fixed coefficient here, but it can be tuned. 
    # Since cost is in [1, 2], and EI/sigma are in objective units, 
    # we want the exploration term to be a modest fraction of EI when EI is large,
    # but significant when EI is small (to keep exploring).
    
    # Heuristic: Exploration weight is 0.5. 
    # This means if EI is 0, the score is 0.5 * sigma / cost.
    # If EI is 10, the score is (10 + 0.5*sigma) / cost.
    exploration_weight = 0.5
    
    numerator = ei + exploration_weight * sigma_safe
    denominator = cost_safe + eps

    # 3. Construct the final score
    # Higher score is better
    scores = numerator / denominator

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if (time.time() - t_start) > 300.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
