"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic LCB with budget-aware exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Estimate remaining budget from current cost level
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5 on average
    # We estimate how much budget remains based on the mean cost seen so far
    # cost is the cost of each candidate, so we use the mean cost to estimate progress
    mean_cost = np.mean(cost_safe)
    
    # Estimate total evaluations so far: if average cost per eval is ~1.5, 
    # and we've spent some budget, we can estimate remaining budget
    # The cost array gives us the cost of each candidate point
    # We don't know exactly how many evals have been done, but we can use
    # the fact that total budget is 30 and initial points cost ~1.5 each
    
    # A better approach: use the cost values to infer budget usage
    # Since we don't have direct access to total spent, we use a heuristic:
    # The cost of points in [0,1]^d is 1 + mean(x_scaled), so costs are in [1, 1+d/2]
    # For 2D: [1, 2], for 3D: [1, 2.5]
    # We'll estimate remaining budget as a fraction based on typical cost
    
    # Use a dynamic beta that decreases as we estimate more budget is used
    # Estimate: if mean cost is ~1.5, and we assume ~20 evals max, budget ~30
    # So remaining budget fraction ~ max(0, 1 - mean_cost * estimated_evals / 30)
    
    # Simpler: use the cost magnitude to scale exploration
    # Lower cost points get more exploration weight
    cost_norm = cost_safe / np.max(cost_safe)
    
    # Dynamic beta: starts high for exploration, decreases for exploitation
    # Use a schedule based on the cost distribution
    # Early on, costs are more uniform; later, we should exploit more
    # Use the spread of costs as a proxy for how "spread out" the search is
    
    # Compute a budget-aware beta
    # If costs are low on average, we've been efficient, so we can explore more
    # If costs are high, we should exploit more
    
    # Beta schedule: beta = beta_max * (1 - progress)
    # where progress is estimated from cost patterns
    
    # Use the ratio of min to max cost as a signal
    # If most points have low cost, we're in a cheap region - explore more
    cost_spread = np.std(cost_safe) / (np.max(cost_safe) + 1e-8)
    
    # Dynamic beta: higher when cost spread is high (exploring diverse regions)
    # Lower when cost spread is low (converging to a region)
    beta_base = 1.5  # Base exploration coefficient
    beta = beta_base * (1.0 + cost_spread)  # More spread = more exploration
    
    # Cost-aware LCB: minimize mu - beta * sigma / cost
    # We want to maximize the score, so: score = -(mu - beta * sigma / cost)
    # = -mu + beta * sigma / cost
    
    # This gives higher scores to points with:
    # - Low predicted mean (good exploitation)
    # - High uncertainty (good exploration)
    # - Low cost (efficient)
    
    scores = -mu + beta * sigma_safe / cost_safe
    
    # Add a small bonus for very low cost points to encourage cheap evaluations
    cost_bonus = 0.1 * (1.0 - cost_norm)
    scores += cost_bonus * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
