import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier selection.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(ei > 0, ei, 0.0)
    
    # 2. Define Objectives for Non-Dominated Sort
    # Objective 1: Maximize EI
    # Objective 2: Maximize Sigma (Uncertainty)
    # Objective 3: Minimize Cost
    
    # We want to find the Pareto front of (EI, Sigma, Cost)
    # Note: Standard non-dominated sort finds points that are not dominated by any other.
    # A point A dominates B if A is better in all objectives and strictly better in at least one.
    # Better: Higher EI, Higher Sigma, Lower Cost.
    
    # Simple O(N^2) non-dominated sort for N=2000 is 4M operations, which is fast in Python/NumPy if vectorized or with simple loops.
    # Let's try to vectorize the dominance check.
    
    # Dominance check:
    # A dominates B if:
    # ei[A] >= ei[B] AND sigma[A] >= sigma[B] AND cost[A] <= cost[B]
    # AND (ei[A] > ei[B] OR sigma[A] > sigma[B] OR cost[A] < cost[B])
    
    # This is hard to vectorize fully for all pairs without memory blowup (2000x2000 matrix is 4M floats, ~32MB, acceptable).
    
    # Let's create a matrix of dominance.
    # D[i, j] is True if i dominates j.
    
    # We can break it down:
    # ei_ge = ei[:, None] >= ei[None, :]
    # sig_ge = sigma[:, None] >= sigma[None, :]
    # cost_le = cost[:, None] <= cost[None, :]
    
    # all_ge_le = ei_ge & sig_ge & cost_le
    # strictly_better = (ei[:, None] > ei[None, :]) | (sigma[:, None] > sigma[None, :]) | (cost[:, None] < cost[None, :])
    
    # dominates = all_ge_le & strictly_better
    
    # is_dominated = np.any(dominates, axis=0) # A point is dominated if any other point dominates it.
    # pareto_mask = ~is_dominated
    
    # Memory check: 3 matrices of 2000x2000 booleans. 4M bytes each. Total ~12MB. Fine.
    
    try:
        ei_ge = ei[:, None] >= ei[None, :]
        sig_ge = sigma[:, None] >= sigma[None, :]
        cost_le = cost[:, None] <= cost[None, :]
        
        all_ge_le = ei_ge & sig_ge & cost_le
        
        # Strictly better in at least one
        # We can reuse the arrays
        strictly_better = (ei[:, None] > ei[None, :]) | (sigma[:, None] > sigma[None, :]) | (cost[:, None] < cost[None, :])
        
        dominates = all_ge_le & strictly_better
        
        # A point is dominated if there exists ANY point that dominates it.
        # dominates[i, j] means i dominates j.
        # We want to know for each j, if any i dominates j.
        # So we check if any row in the column j is True.
        # np.any(dominates, axis=0) gives a boolean array of length 2000.
        # If True, the point is dominated.
        
        is_dominated = np.any(dominates, axis=0)
        pareto_mask = ~is_dominated
        
    except MemoryError:
        # Fallback to simple scalarization if memory fails (unlikely)
        pareto_mask = np.ones(len(ei), dtype=bool)
        
    # 3. Score Assignment
    # Points on the Pareto front are candidates.
    # Among them, we need to pick the best one.
    # We use EI / Cost as a tie-breaker/selection metric for the Pareto points.
    # Non-Pareto points get 0.
    
    scores = np.zeros_like(ei)
    
    if np.any(pareto_mask):
        # Calculate efficiency for Pareto points
        # We want to maximize EI/Cost.
        # To avoid division by zero, cost is already floored.
        efficiency = ei / cost
        efficiency[~pareto_mask] = 0.0
        
        # Normalize efficiency to [0, 1] for stable scores?
        # Or just use raw efficiency.
        # The evaluator picks argmax.
        # If we just return efficiency, non-Pareto points are 0.
        # Pareto points have positive efficiency (since EI > 0 for most, and cost > 0).
        
        # However, if all EI are 0, scores are 0.
        
        scores = efficiency
        
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
