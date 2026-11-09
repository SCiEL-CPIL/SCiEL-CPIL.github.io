"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # 30s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto filtering and multiplicative hypervolume scoring.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # We want to maximize EI (improvement) and minimize cost.
    # Pareto front in (ei, -cost) space: candidate A dominates B if ei_A >= ei_B and cost_A <= cost_B.
    # Since we want to select from 2000 candidates, we identify the Pareto front.
    
    # To find the Pareto front efficiently:
    # 1. Sort by EI descending.
    # 2. Iterate, keeping track of the minimum cost seen so far.
    # 3. If current cost < min_cost_so_far, it's on the front (or at least not dominated by previous higher-EI points).
    
    n = len(mu)
    # Sort indices by EI descending
    ei_sorted_idx = np.argsort(-ei)
    
    # Initialize scores to 0
    scores = np.zeros(n)
    
    # Find Pareto front
    # A point is on the Pareto front if no other point has both higher EI and lower cost.
    # Since we sorted by EI descending, for each point in sorted order, if its cost is less than 
    # the minimum cost of all points with higher (or equal) EI, it is on the front.
    
    min_cost_so_far = np.inf
    front_mask = np.zeros(n, dtype=bool)
    
    for idx in ei_sorted_idx:
        if cost[idx] < min_cost_so_far:
            front_mask[idx] = True
            min_cost_so_far = cost[idx]
    
    # For points on the Pareto front, compute a hypervolume-like score.
    # The idea: score = EI * (1/cost) or similar. But the inspiration says "multiplicative hypervolume scoring (EI * cost_savings)".
    # "Cost savings" is ambiguous. Let's interpret it as the benefit relative to cost.
    # A common cost-aware EI is EI / cost. The inspiration suggests EI * (something).
    # Let's use EI / cost for the front points, but perhaps scaled or modified.
    # Actually, the cell finding says "multiplicative hypervolume scoring (EI * cost_savings) better balances...".
    # If cost_savings is defined as (max_cost - cost) or 1/cost?
    # Let's stick to EI / cost for the front points, as it's a standard cost-aware metric, 
    # but the *filtering* is the structural change.
    # Wait, the inspiration specifically mentions "multiplicative hypervolume scoring (EI * cost_savings)".
    # Let's try: score = EI * (1.0 / cost). This is equivalent to EI / cost.
    # Why would it be different? Maybe the filtering removes dominated points that have high EI but high cost,
    # allowing the optimizer to pick a point with moderate EI but very low cost, which EI/cost might not highlight as strongly 
    # if the EI is very high.
    
    # Let's compute EI / cost for all points, but set dominated points to 0.
    # This forces the selection to be from the Pareto front.
    
    ei_over_cost = ei / cost
    
    # Only keep scores for Pareto-optimal points
    scores[front_mask] = ei_over_cost[front_mask]
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
