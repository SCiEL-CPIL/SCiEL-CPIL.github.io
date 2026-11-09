# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a linear cost penalty model.
    Score = (EI / scale) - (penalty * cost) + (exploration * sigma / scale)
    
    This structure avoids the over-penalization of expensive regions seen in
    ratio-based methods (EI/cost) while maintaining cost awareness.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    # For minimization: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in CDF/PDF
    z = np.clip(z, -30, 30)
    
    # Compute EI (Expected Improvement for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive scale: Normalizes EI and exploration bonus relative to the current best value.
    # This makes the score invariant to the magnitude of the objective function.
    scale = max(abs(f_best), 1e-6)
    
    # Penalty for cost.
    # A linear penalty allows high-EI points to overcome cost if the improvement is significant.
    # 0.1 is a heuristic balance: 1 unit of cost is "worth" 0.1 units of normalized improvement.
    penalty = 0.1
    
    # Exploration bonus: encourages sampling in high-uncertainty regions,
    # especially when the best value is small (late in optimization).
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / scale
    
    # Combined score: higher is better
    # Normalized EI minus linear cost penalty plus exploration bonus
    scores = (ei / scale) - (penalty * cost_safe) + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
