import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _compute_radius(centres):
    """Compute the actual packing radius for given centres."""
    if len(centres) < 2:
        return 0.0
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(len(centres), k=1)
    d_min = np.sqrt(dist_sq[iu].min())
    
    # Distance to boundary
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    b_min = np.minimum(0.5 - dx, 0.5 - dy).min()
    
    if b_min < 0:
        return -np.inf
        
    return min(d_min / 2.0, b_min)

def _force_directed_energy(pts, target_r):
    """
    Force-directed energy: repel pairs closer than 2*target_r, 
    penalize boundary violations.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    
    # Repulsion: only for pairs closer than 2*target_r
    # Use a smooth potential: 1/d^2 for d < 2r, 0 otherwise
    # Actually, let's use a standard Lennard-Jones-like or just inverse square for close pairs.
    # To be efficient and smooth, let's use:
    # E_rep = sum (max(0, (2r - d)^2)) ? No, that's soft.
    # Let's use sum 1/d^2 for d < 2r, and a large penalty if d is very small.
    # Better: minimize sum (1/d^p) for all pairs, but that's what the old code did.
    
    # Let's try a different approach: 
    # Maximize r such that all constraints are satisfied.
    # This is hard to do directly with gradient descent on r.
    
    # Instead, let's use the objective:
    # Minimize sum_{i<j} max(0, (2r - d_ij)^2) + sum_i max(0, (r - b_i)^2)
    # This is a smooth approximation of the constraint violation.
    
    # But we want to find the best r. 
    # A common trick is to fix r and check feasibility, or optimize a combined objective.
    
    # Let's stick to a repulsion potential that is strong for close pairs.
    # E = sum_{i<j} (1/d_ij^2) + K * sum_i (boundary_violation_i^2)
    # But this doesn't directly maximize the minimum distance.
    
    # Alternative: 
    # Minimize sum_{i<j} (1/d_ij^p) + sum_i (1/b_i^p)
    # As p increases, this approaches the minimax problem.
    
    # Let's use p=4 for a sharper focus on the minimum distance.
    
    d_sq_clipped = np.clip(d_sq, 1e-8, None)
    repulsion = np.sum(1.0 / (d_sq_clipped ** 2.0))
    
    # Boundary
    dx = np.abs(pts[:, 0])
    dy = np.abs(pts[:, 1])
    b = np.minimum(0.5 - dx, 0.5 - dy)
    
    # Penalty for being outside or close to boundary
    # If b < 0, it's outside. 
    # We want to maximize min(b). 
    # So minimize sum(1/b^2) for b > 0, and large penalty for b <= 0.
    
    b_clipped = np.clip(b, 1e-8, None)
    boundary_repulsion = np.sum(1.0 / (b_clipped ** 2.0))
    
    # Add hard penalty for outside
    outside = np.maximum(0, -b)
    hard_penalty = 100.0 * np.sum(outside**2)
    
    return repulsion + boundary_repulsion + hard_penalty

def _soft_min_optimize(pts, target_r, time_limit):
    """
    Try to optimize the layout to achieve radius target_r.
    Returns the best points found.
    """
    bounds = [(-0.5, 0.5)] * (2 * len(pts))
    
    # We will minimize a function that penalizes violations of the target radius.
    # E = sum_{i<j} max(0, (2*target_r - d_ij)^2) + sum_i max(0, (target_r - b_i)^2)
    # This is differentiable almost everywhere.
    
    def _violation_energy(x):
        p = x.reshape(-1, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(len(p), k=1)
        d = np.sqrt(dist_sq[iu])
        
        # Pair violations
        pair_viol = np.maximum(0, 2*target_r - d)
        E_pair = np.sum(pair_viol**2)
        
        # Boundary violations
        dx = np.abs(p[:, 0])
        dy = np.abs(p[:, 1])
        b = np.minimum(0.5 - dx, 0.5 - dy)
        b_viol = np.maximum(0, target_r - b)
        E_bound = np.sum(b_viol**2)
        
        return E_pair + E_bound
    
    # Start from current pts
    x0 = pts.flatten()
    
    try:
        result = minimize(
            _violation_energy, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        new_pts = result.x.reshape(-1, 2)
        return new_pts
    except:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial layout: Hexagonal grid
    # Estimate spacing
    # Area = 1, N=200. Area/point = 0.005.
    # Hex cell area ~ s^2 * sqrt(3)/2.
    # s ~ sqrt(0.005 * 2 / sqrt(3)) ~ 0.076
    
    s = 0.076
    h = s * np.sqrt(3) / 2
    
    candidates = []
    max_dim = 30
    for i in range(-max_dim, max_dim):
        for j in range(-max_dim, max_dim):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
                
    candidates = np.array(candidates)
    
    if len(candidates) < N:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        candidates = grid[:N]
    else:
        # Select N points
        # Random selection to avoid bias? Or central?
        # Let's try a stratified selection or just random.
        idx = np.random.choice(len(candidates), N, replace=False)
        candidates = candidates[idx]
        
    best_points = candidates.copy()
    best_r = _compute_radius(best_points)
    
    # Iterative improvement: 
    # 1. Estimate current best r.
    # 2. Try to increase r by a small factor.
    # 3. Run soft-min optimization to find a layout for the new r.
    # 4. If successful (violation ~ 0), update best_r and best_points.
    # 5. If time runs out, return best_points.
    
    current_r = best_r
    # Increase r gradually
    # Start with a small increase to avoid getting stuck
    for step in range(50):
        if time.time() > deadline:
            break
            
        # Try to increase r by 1%
        new_r = current_r * 1.01
        # Also try a slightly larger jump every 10 steps?
        
        # Run optimization
        new_pts = _soft_min_optimize(best_points, new_r, deadline - time.time())
        
        # Check if the new layout achieves new_r
        actual_r = _compute_radius(new_pts)
        
        if actual_r >= new_r * 0.99:  # Allow small numerical error
            # Success! Update best
            best_points = new_pts
            current_r = actual_r
            best_r = actual_r
        else:
            # Failed to achieve new_r. 
            # Maybe try a smaller increase?
            # For now, just break or try again with same r?
            # Let's try to refine at the current best_r to see if we can get a better layout
            # that might allow a larger r later.
            
            # Or just stop increasing r and try to polish the current layout.
            # Let's try to polish the current best_points with the current_r.
            polished_pts = _soft_min_optimize(best_points, current_r, deadline - time.time())
            polished_r = _compute_radius(polished_pts)
            
            if polished_r > best_r:
                best_points = polished_pts
                best_r = polished_r
                current_r = polished_r
            else:
                # If we can't even maintain current_r, or improve, maybe we are stuck.
                # Let's try a different initialization?
                # Or just break.
                break
                
    return best_points
