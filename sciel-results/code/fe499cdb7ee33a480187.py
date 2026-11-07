"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Lower Confidence Bound with dynamic kappa.
    # kappa increases as budget is consumed to shift from exploration to exploitation.
    # Total budget is 30. We track progress via mean cost of evaluated points.
    # cost array contains the cost of each candidate (not yet evaluated), 
    # but we can estimate budget consumption from the current state.
    # Since we don't have direct access to total spent, we use a heuristic:
    # The cost array values are in [1, 2]. 
    # We estimate remaining budget fraction based on how many evaluations have occurred.
    # However, the signature doesn't give us the number of evaluations.
    # Instead, we can use the magnitude of f_best or sigma to gauge convergence.
    # A better approach: use a fixed schedule based on the cost values themselves.
    # Low cost points are "cheap" so we can afford more exploration there.
    # High cost points are "expensive" so we should be more exploitative.
    
    # Let's use a dynamic kappa that depends on the cost of each candidate.
    # Candidates with higher cost get a higher kappa (more exploitation).
    # Candidates with lower cost get a lower kappa (more exploration).
    
    # Normalize cost to [0, 1] for kappa calculation.
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Base kappa starts at 1.0 and increases with cost_norm.
    # Also, we want to shift from exploration to exploitation as f_best improves.
    # If f_best is close to 0 (the minimum), we should be more exploitative.
    # Let's use a combination.
    
    # Heuristic for global progress: if f_best is small, we're doing well.
    # For Griewank, the minimum is 0.
    # Let's say if f_best < 0.1, we're in exploitation mode.
    # This is a rough heuristic.
    
    # A simpler and more robust approach:
    # kappa = base_kappa + cost_sensitivity * cost_norm
    # Let's try base_kappa = 1.0, cost_sensitivity = 1.0
    # So kappa ranges from 1.0 to 2.0 depending on cost.
    
    kappa = 1.0 + 1.0 * cost_norm
    
    # Also, we can make kappa larger if f_best is very small (near optimum).
    # But we don't know the scale of f_best without prior knowledge.
    # Let's stick with cost-based kappa for now.
    
    # LCB: mu - kappa * sigma
    # We want to maximize this, so we score as:
    # score = (mu - kappa * sigma) / cost
    # But wait, the task says "higher = evaluate next".
    # And we want to minimize the function.
    # So low mu is good.
    # LCB = mu - kappa * sigma. We want to maximize LCB/cost?
    # No, LCB is a lower bound on the true value. We want the lowest possible value.
    # So we want to minimize LCB.
    # But the acquisition function should return higher scores for better candidates.
    # So we should return -LCB / cost or (f_best - LCB) / cost?
    
    # Let's reconsider.
    # In minimization, we want to find points with low f(x).
    # LCB = mu - kappa * sigma is a lower bound on f(x).
    # If LCB is low, it's a good candidate.
    # So we want to minimize LCB.
    # But the return value should be higher for better candidates.
    # So we should return -LCB / cost.
    
    # Alternatively, we can use Expected Improvement (EI) which is already positive for improvement.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is positive when mu < f_best.
    # We can divide by cost: EI / cost.
    
    # The previous program used EI / cost and scored 1.0.
    # The inspiration suggests LCB: (mu - kappa * sigma) / cost.
    # But if we maximize (mu - kappa * sigma) / cost, we are maximizing mu, which is bad for minimization.
    # So it should be (-(mu - kappa * sigma)) / cost = (kappa * sigma - mu) / cost.
    # Or simply - (mu - kappa * sigma) / cost.
    
    # Let's use: score = (kappa * sigma - mu) / cost
    # This rewards high uncertainty (sigma) and low mean (mu), penalized by cost.
    
    score = (kappa * sigma - mu) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
