"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Knowledge Gradient (KG) with adaptive exploration.
    KG estimates expected reduction in regret, providing optimal exploration/exploitation.
    We blend KG with a cost-aware exploration bonus to balance information gain and cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 330s limit, return a simple safe result
    if time.time() - _start_time > 310:
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -10, 10)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0.0)
        scores = EI / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Knowledge Gradient (KG): expected reduction in regret
    # KG(x) = E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is equivalent to EI but interpreted as expected regret reduction
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    kg = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    kg = np.where(np.isfinite(kg), kg, 0.0)
    kg = np.maximum(kg, 0.0)
    
    # Adaptive exploration bonus: reward uncertainty in regions where KG is low
    # This encourages exploration when exploitation (KG) is not promising
    # Use a soft bonus that scales with sigma and decreases as KG increases
    exploration_weight = 0.3
    exploration_bonus = exploration_weight * sigma * np.exp(-kg / (sigma + 1e-9))
    exploration_bonus = np.where(np.isfinite(exploration_bonus), exploration_bonus, 0.0)
    
    # Combine KG and exploration bonus
    combined = kg + exploration_bonus
    combined = np.where(np.isfinite(combined), combined, 0.0)
    combined = np.maximum(combined, 0.0)
    
    # Cost-aware scaling: divide by cost to prefer cheaper evaluations
    # Use a softer cost penalty: divide by (1 + cost) instead of cost
    # This avoids extreme penalization of expensive points while still preferring cheap ones
    cost_factor = 1.0 + cost
    scores = combined / cost_factor
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
