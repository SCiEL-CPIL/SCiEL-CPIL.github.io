import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Normalized Expected Improvement (EI) combined with Normalized Exploration (UCB-like),
    penalized by the inverse of the evaluation cost.
    
    This structure ensures that the exploitation (EI) and exploration (uncertainty) terms
    are on comparable scales, and the cost penalty strongly favors cheaper evaluations
    to maximize the number of samples within the fixed budget.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1] to balance with exploration term
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # Calculate Normalized Exploration Term
    # Scale-invariant uncertainty measure
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        exploration = sigma / sigma_max
    else:
        exploration = np.zeros_like(sigma)
        
    # Combine Exploitation and Exploration
    # Beta = 1.0 balances the two normalized terms equally.
    # This allows the optimizer to consider both promising areas (low mu)
    # and uncertain areas (high sigma).
    beta = 1.0
    base_score = ei_norm + beta * exploration
    
    # Apply Cost Penalty
    # Use linear inverse cost (1/cost) as suggested by effective strategies for tight budgets.
    # Cost is in [1, 2]. 1/cost maps [1, 2] to [1.0, 0.5].
    # This strongly penalizes expensive evaluations, ensuring we spend the budget
    # on more, cheaper evaluations which is often more effective for convergence
    # in BO with limited budgets.
    cost_penalty = 1.0 / cost
    
    scores = base_score * cost_penalty
    
    # Deadline guard
    if time.time() - t_start > 2.0:
        # Fallback to exploration if computation is unexpectedly slow
        return exploration * cost_penalty
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
