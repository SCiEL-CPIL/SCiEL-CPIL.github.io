"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering and hypervolume contribution.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: scalar, lowest observed objective value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of scores (higher = evaluate next).
    """
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute expected improvement (EI) for each candidate
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Compute cost-aware score: we want to maximize improvement per unit cost,
    # but use Pareto-frontier filtering to retain only non-dominated points.
    # 
    # Multi-objective: maximize EI, minimize cost.
    # A point (ei_i, cost_i) dominates (ei_j, cost_j) if ei_i >= ei_j AND cost_i <= cost_j,
    # with at least one strict inequality.
    #
    # First, compute the scalar cost-aware score for all points:
    # score_raw = ei / cost
    cost_safe = np.maximum(cost, 1e-12)
    score_raw = ei / cost_safe
    
    # Build the Pareto frontier: keep points that are not dominated by any other point.
    # Dominance: i dominates j if ei[i] >= ei[j] AND cost[i] <= cost[j] AND (ei[i] > ei[j] OR cost[i] < cost[j])
    #
    # For efficiency with 2000 points, we can sort and use a sweep.
    # Sort by cost ascending, then for equal cost, sort by EI descending.
    # A point is on the Pareto frontier if its EI is greater than all previously seen points with lower or equal cost.
    
    # Sort indices by cost (ascending), break ties by EI (descending)
    order = np.lexsort((-ei, cost))
    
    # Track the maximum EI seen so far as we sweep through sorted points
    is_pareto = np.zeros(n, dtype=bool)
    max_ei_so_far = -np.inf
    
    for idx in order:
        if ei[idx] > max_ei_so_far + 1e-15:
            is_pareto[idx] = True
            max_ei_so_far = ei[idx]
        # If ei[idx] <= max_ei_so_far, it's dominated by some earlier point (lower cost, higher EI)
    
    # For non-Pareto points, assign a low score (but still finite and positive if possible)
    # For Pareto points, compute hypervolume contribution.
    
    # Hypervolume contribution: For each Pareto point, compute the area of the rectangle
    # it contributes to the hypervolume with respect to the reference point (0, max_cost) 
    # or more appropriately, we compute the contribution relative to the current best.
    #
    # In the (improvement, cost) space, we want to maximize improvement and minimize cost.
    # Let's define the reference point as (0, cost_max) where cost_max is the maximum cost.
    # The hypervolume of a point (ei_i, cost_i) relative to reference (0, cost_max) is:
    # HV_i = ei_i * (cost_max - cost_i)
    # But this isn't quite right for hypervolume contribution in the Pareto sense.
    #
    # A better approach: the hypervolume contribution of point i is the area of the region
    # that is dominated by point i but not by any other Pareto point.
    #
    # For simplicity and computational efficiency, let's use a simpler metric:
    # For Pareto points, score = ei_i * (1 / cost_i) weighted by the "gap" to the next Pareto point.
    #
    # Actually, let's use a cleaner approach: 
    # Score for Pareto points = EI * (1 - cost / cost_max) / cost
    # This rewards high EI and low cost.
    
    cost_max = np.max(cost)
    
    # For Pareto points, compute a hypervolume-inspired score
    # The hypervolume contribution of point i in the 2D case (maximize EI, minimize cost)
    # with reference point (0, cost_max) can be approximated as:
    # Each Pareto point contributes a trapezoidal area.
    
    # Get Pareto points sorted by cost
    pareto_indices = np.where(is_pareto)[0]
    
    if len(pareto_indices) == 0:
        # Fallback: use raw score
        return np.maximum(score_raw, 1e-12)
    
    # Sort Pareto indices by cost
    pareto_sorted = pareto_indices[np.argsort(cost[pareto_indices])]
    
    # Compute hypervolume contribution for each Pareto point
    # For a set of Pareto points sorted by cost (ascending), the hypervolume contribution
    # of point i is: ei_i * (cost_{i+1} - cost_i) where cost_{n} = cost_max
    # (this is the area of the rectangle from (cost_i, 0) to (cost_{i+1}, ei_i) that is uniquely dominated by point i)
    
    scores = np.zeros(n, dtype=np.float64)
    
    for i, idx in enumerate(pareto_sorted):
        if i < len(pareto_sorted) - 1:
            next_cost = cost[pareto_sorted[i + 1]]
        else:
            next_cost = cost_max
        
        # Width in cost dimension
        width = next_cost - cost[idx]
        # If width is negative or zero, use a small value
        width = max(width, 1e-12)
        
        # Hypervolume contribution
        hv_contrib = ei[idx] * width
        
        # Also incorporate the raw cost-efficiency
        scores[idx] = hv_contrib + ei[idx] / cost_safe[idx]
    
    # For non-Pareto points, give them a score slightly below the minimum Pareto score
    min_pareto_score = np.min(scores[is_pareto]) if np.any(is_pareto) else 0.0
    non_pareto_mask = ~is_pareto
    if np.any(non_pareto_mask):
        # Give non-Pareto points a fraction of their raw score, scaled down
        scores[non_pareto_mask] = score_raw[non_pareto_mask] * 0.5
    
    # Add a small exploration bonus for high uncertainty (sigma) to encourage exploration
    # This helps escape local optima in Ackley function
    exploration_bonus = 0.1 * sigma_safe
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are non-negative
    scores = np.maximum(scores, 0.0)
    
    # Add a small epsilon to avoid all-zero scores
    scores = scores + 1e-12
    
    return scores
# EVOLVE-BLOCK-END
