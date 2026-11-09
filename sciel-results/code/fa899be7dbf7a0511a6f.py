"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering and hypervolume contribution scoring.
    This replaces scalar division by cost with a multi-objective utility approach:
    1. Compute expected improvement (EI) for each candidate.
    2. Identify the Pareto front of (EI, cost) where higher EI and lower cost are better.
    3. Score each candidate by its hypervolume contribution relative to the nadir point.
    
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
    
    # Compute Expected Improvement (EI) as the primary improvement metric
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Also compute a UCB-like exploration bonus to encourage visiting uncertain regions
    # This helps with the multi-objective front by giving some credit to high-sigma points
    beta = 1.0
    ucb_bonus = beta * sigma
    ucb_bonus = np.where(np.isfinite(ucb_bonus), ucb_bonus, 0.0)
    
    # Combined objective: EI + small exploration bonus
    # This ensures that even points with zero expected improvement but high uncertainty
    # can appear on the Pareto front if their cost is sufficiently low
    obj = ei + 0.1 * ucb_bonus
    obj = np.where(np.isfinite(obj), obj, 0.0)
    
    # Pareto frontier identification in 2D:
    # A point is on the Pareto front if no other point has both higher obj AND lower cost
    # We use a vectorized approach:
    # For each point i, it is dominated if there exists j such that obj[j] >= obj[i] and cost[j] <= cost[i]
    # and at least one inequality is strict.
    
    # Efficient 2D Pareto filter:
    # Sort by obj descending, then keep track of minimum cost seen so far
    order = np.argsort(-obj)  # descending order of obj
    sorted_obj = obj[order]
    sorted_cost = cost[order]
    
    # A point is on the Pareto front if its cost is less than all costs of points 
    # with higher (or equal) obj that come before it in the sorted order
    # More precisely: point at position i in sorted order is on front if 
    # sorted_cost[i] < min(sorted_cost[0:i]) (strictly less than all previous)
    
    n = len(obj)
    on_front = np.zeros(n, dtype=bool)
    
    if n > 0:
        # For the first point (highest obj), it's on the front
        on_front[order[0]] = True
        min_cost_so_far = sorted_cost[0]
        for i in range(1, n):
            if sorted_cost[i] < min_cost_so_far:
                on_front[order[i]] = True
                min_cost_so_far = sorted_cost[i]
    
    # Hypervolume contribution for each Pareto point:
    # For 2D with objectives (obj, -cost) [we want to maximize obj and minimize cost],
    # the hypervolume contribution of a point is the area of the rectangle it uniquely covers.
    # 
    # The reference point (nadir) is (min_obj, max_cost) among all points, or we can use (0, max_cost).
    # For simplicity, use (0, max_cost) as reference since obj >= 0.
    
    # Actually, let's compute the hypervolume contribution properly:
    # Sort Pareto points by obj descending (and cost ascending along the front)
    front_indices = np.where(on_front)[0]
    if len(front_indices) > 0:
        front_obj = obj[front_indices]
        front_cost = cost[front_indices]
        
        # Sort front points by obj descending
        front_order = np.argsort(-front_obj)
        front_obj_sorted = front_obj[front_order]
        front_cost_sorted = front_cost[front_order]
        
        # Reference point: (0, max_cost) where max_cost is the maximum cost in the entire set
        ref_obj = 0.0
        ref_cost = np.max(cost)
        
        # Hypervolume contribution of each point on the front:
        # For point i (in sorted order), its contribution is:
        # (front_obj_sorted[i] - next_obj) * (ref_cost - front_cost_sorted[i])
        # where next_obj is the obj of the next point (or ref_obj for the last)
        
        hv_contrib = np.zeros(n, dtype=np.float64)
        
        for i in range(len(front_obj_sorted)):
            current_obj = front_obj_sorted[i]
            current_cost = front_cost_sorted[i]
            
            # Width in obj dimension: from current_obj to next point's obj (or ref_obj)
            if i + 1 < len(front_obj_sorted):
                next_obj = front_obj_sorted[i + 1]
            else:
                next_obj = ref_obj
            width_obj = current_obj - next_obj
            
            # Height in cost dimension: from current_cost to ref_cost
            height_cost = ref_cost - current_cost
            
            contrib = width_obj * height_cost
            if contrib > 0:
                # Map back to original index
                orig_idx = front_indices[front_order[i]]
                hv_contrib[orig_idx] = contrib
    
    # For points not on the Pareto front, assign a lower score based on their 
    # proximity to the front. We use a decay based on how much they are dominated.
    # Simple approach: for dominated points, score = 0 or a small fraction.
    # Alternatively, we can use the ratio of their potential contribution.
    
    # Let's use a simpler scoring: Pareto points get their hypervolume contribution,
    # dominated points get a score proportional to how close they are to being on the front.
    
    # For dominated points, compute the "domination gap":
    # How much would obj need to increase (or cost decrease) to be on the front?
    
    scores = np.zeros(n, dtype=np.float64)
    
    # Assign hypervolume contributions to front points
    scores[on_front] = hv_contrib[on_front]
    
    # For dominated points, assign a small score based on their obj/cost ratio
    # This ensures they can still be selected if all front points have very low scores
    dominated = ~on_front
    if np.any(dominated):
        # Normalize obj and cost to [0, 1] for a simple scoring
        obj_min, obj_max = np.min(obj), np.max(obj)
        cost_min, cost_max = np.min(cost), np.max(cost)
        
        if obj_max > obj_min:
            obj_norm = (obj - obj_min) / (obj_max - obj_min)
        else:
            obj_norm = np.ones(n)
        
        if cost_max > cost_min:
            cost_norm = 1.0 - (cost - cost_min) / (cost_max - cost_min)
        else:
            cost_norm = np.ones(n)
        
        # Score for dominated points: small weight times normalized obj * normalized cost benefit
        dom_scores = 0.05 * obj_norm * cost_norm
        scores[dominated] = dom_scores[dominated]
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Normalize scores to [0, 1] range for consistency
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    
    return scores
# EVOLVE-BLOCK-END
