import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin from 330s limit
    
    n = len(mu)
    scores = np.zeros(n)
    
    # Hypervolume-based acquisition for minimization
    # We work in the space where lower mu and lower sigma are better.
    # Reference point: (f_best, max_sigma) where max_sigma is a large value
    # Actually, for minimization, we want to minimize both mu and sigma.
    # The Pareto front consists of non-dominated (mu, sigma) pairs.
    # A point dominates another if it has <= mu and <= sigma, with at least one strict.
    
    # For hypervolume in minimization, we can transform to maximization:
    # Let u = -mu, v = -sigma. Then maximize u and v.
    # Reference point in (u,v) space: (u_ref, v_ref) where u_ref = -f_best (or lower), v_ref = -max_sigma
    
    # Simpler approach: compute for each candidate the improvement it would bring
    # to the set of non-dominated points, measured by hypervolume.
    
    # Step 1: Identify non-dominated points among the candidates
    # A point i is dominated by point j if mu[j] <= mu[i] and sigma[j] <= sigma[i]
    # with at least one strict inequality.
    
    # For efficiency, we'll use a vectorized approach to find non-dominated points
    # and compute their contribution to hypervolume.
    
    # First, let's find which points are non-dominated
    # Point i is non-dominated if there's no point j with mu[j] <= mu[i] and sigma[j] <= sigma[i]
    # and (mu[j] < mu[i] or sigma[j] < sigma[i])
    
    # This is O(n^2) which is too slow for n=2000.
    # Instead, we'll use a different strategy:
    
    # Sort by mu, then for each point, check if any earlier point has smaller sigma.
    # If we sort by mu ascending, a point is dominated if there exists a previous point
    # with sigma <= current sigma.
    
    mu_sorted_idx = np.argsort(mu)
    mu_sorted = mu[mu_sorted_idx]
    sigma_sorted = sigma[mu_sorted_idx]
    
    # For each point in sorted order, it's non-dominated if its sigma is less than
    # the minimum sigma of all points with smaller mu.
    min_sigma_so_far = np.full(n, np.inf)
    min_sigma_so_far[0] = sigma_sorted[0]
    for i in range(1, n):
        min_sigma_so_far[i] = min(min_sigma_so_far[i-1], sigma_sorted[i])
    
    # Point i (in sorted order) is non-dominated if sigma_sorted[i] < min_sigma_so_far[i-1]
    # (for i > 0), or always for i = 0.
    non_dominated_mask_sorted = np.zeros(n, dtype=bool)
    non_dominated_mask_sorted[0] = True
    for i in range(1, n):
        non_dominated_mask_sorted[i] = sigma_sorted[i] < min_sigma_so_far[i-1]
    
    # Map back to original indices
    non_dominated_mask = np.zeros(n, dtype=bool)
    non_dominated_mask[mu_sorted_idx] = non_dominated_mask_sorted
    
    # Now, for non-dominated points, compute their hypervolume contribution.
    # In the (mu, sigma) space with minimization, the hypervolume of a point
    # relative to a reference point is the area of the rectangle it "controls".
    
    # Reference point: (f_best, sigma_ref) where sigma_ref is large enough.
    # Actually, for a meaningful hypervolume, we need a reference point that is
    # worse than all points. Let's use (f_best + 1, max(sigma) + 1) as reference.
    
    # But this is getting complex. Let's use a simpler but effective approach:
    # Score each point by a combination of:
    # 1. How much it improves upon f_best (exploitation)
    # 2. Its uncertainty (exploration)
    # 3. Whether it's on the Pareto front (diversity)
    
    # Actually, let's implement a proper hypervolume-based score.
    # For each non-dominated point, its hypervolume contribution is the area of the
    # rectangle from that point to the reference point, minus contributions of
    # other non-dominated points that "cover" part of it.
    
    # Simplified: For minimization, the hypervolume of the Pareto front is the area
    # under the curve when sorted by mu. Each non-dominated point contributes a
    # rectangle of width (next_mu - current_mu) and height (sigma_ref - current_sigma).
    
    # Let's get the non-dominated points sorted by mu
    nd_indices = np.where(non_dominated_mask)[0]
    nd_mu = mu[nd_indices]
    nd_sigma = sigma[nd_indices]
    
    # Sort by mu
    sort_idx = np.argsort(nd_mu)
    nd_mu_sorted = nd_mu[sort_idx]
    nd_sigma_sorted = nd_sigma[sort_idx]
    nd_indices_sorted = nd_indices[sort_idx]
    
    # Reference point
    mu_ref = f_best + 1.0  # worse than f_best
    sigma_ref = np.max(sigma) + 1.0  # worse than max sigma
    
    # Compute hypervolume contribution for each non-dominated point
    # The contribution of point i is the area of the rectangle it uniquely covers.
    # For the last point (largest mu), it covers from nd_mu_sorted[-1] to mu_ref
    # with height (sigma_ref - nd_sigma_sorted[-1]).
    # For other points, it covers from nd_mu_sorted[i] to nd_mu_sorted[i+1]
    # with height (sigma_ref - nd_sigma_sorted[i]).
    
    hv_contributions = np.zeros(len(nd_mu_sorted))
    for i in range(len(nd_mu_sorted)):
        if i < len(nd_mu_sorted) - 1:
            width = nd_mu_sorted[i+1] - nd_mu_sorted[i]
        else:
            width = mu_ref - nd_mu_sorted[i]
        height = sigma_ref - nd_sigma_sorted[i]
        if width > 0 and height > 0:
            hv_contributions[i] = width * height
    
    # Assign hypervolume contributions to scores
    scores[nd_indices_sorted] = hv_contributions
    
    # For dominated points, give them a small score based on how close they are
    # to the Pareto front, to encourage exploration in promising regions.
    dominated_mask = ~non_dominated_mask
    if np.any(dominated_mask):
        # Score dominated points by their potential to become non-dominated
        # if sigma decreases. Use a simple heuristic: score = f_best - mu, but
        # only if mu < f_best.
        dom_mu = mu[dominated_mask]
        dom_sigma = sigma[dominated_mask]
        
        # Base score: how much improvement over f_best
        base_score = np.maximum(f_best - dom_mu, 0.0)
        
        # Bonus for high uncertainty (potential to improve)
        uncertainty_bonus = dom_sigma * 0.5
        
        scores[dominated_mask] = base_score + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
