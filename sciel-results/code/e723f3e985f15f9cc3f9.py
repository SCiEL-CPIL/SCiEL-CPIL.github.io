"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Pareto-frontier-inspired scoring.
    
    Combines Expected Improvement (EI) with uncertainty and cost considerations.
    Uses a rank-based efficiency filter to balance exploitation and exploration
    without arbitrary weight tuning.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute multiple objectives for Pareto consideration
    # 1. Expected improvement (higher is better)
    obj_improvement = ei
    
    # 2. Uncertainty (higher sigma means more uncertain, potentially valuable)
    obj_uncertainty = sigma_safe
    
    # 3. Inverse cost (lower cost is better, so we use 1/cost)
    obj_cost = 1.0 / cost_safe
    
    # Normalize each objective to [0, 1] range for fair comparison
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_improvement = normalize(obj_improvement)
    norm_uncertainty = normalize(obj_uncertainty)
    norm_cost = normalize(obj_cost)
    
    # Check time
    if time.time() > deadline - 10.0:
        # Return simple cost-aware EI
        scores = ei / np.sqrt(cost_safe)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute non-domination ranks (simplified Pareto front)
    # A point dominates another if it's better or equal in all objectives
    # and strictly better in at least one
    n = len(mu)
    
    # Use a simplified approach: compute a weighted sum that approximates Pareto efficiency
    # Weight based on how "extreme" each candidate is in each dimension
    
    # For each candidate, compute how many other candidates dominate it
    # This is O(n^2) which is 4M comparisons for n=2000, acceptable
    
    # Instead, use a faster heuristic: compute the rank in each dimension
    # and combine them
    
    rank_improvement = np.argsort(np.argsort(-obj_improvement))  # Lower rank = better
    rank_uncertainty = np.argsort(np.argsort(-obj_uncertainty))
    rank_cost = np.argsort(np.argsort(-obj_cost))  # Lower rank = lower cost
    
    # Normalize ranks to [0, 1]
    rank_improvement_norm = rank_improvement / max(n - 1, 1)
    rank_uncertainty_norm = rank_uncertainty / max(n - 1, 1)
    rank_cost_norm = rank_cost / max(n - 1, 1)
    
    # A candidate is on the Pareto front if it has a good rank in at least one dimension
    # and is not dominated in all dimensions
    # Approximate: sum of normalized ranks, lower is better (closer to Pareto front)
    pareto_score = rank_improvement_norm + rank_uncertainty_norm + rank_cost_norm
    
    # Invert so that better Pareto position gives higher score
    pareto_efficiency = 1.0 - pareto_score / 3.0
    
    # Combine EI with Pareto efficiency
    # Use a blend: primarily EI, boosted by Pareto efficiency
    # This rewards candidates that are efficient in the multi-objective sense
    
    # Base score is cost-adjusted EI
    base_score = ei / np.sqrt(cost_safe)
    
    # Boost by Pareto efficiency
    # Candidates on the Pareto front get a boost
    boost = 0.3 * pareto_efficiency
    
    scores = base_score * (1.0 + boost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
