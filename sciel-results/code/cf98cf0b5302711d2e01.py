"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 50

# EVOLVE-BLOCK-START

def _calculate_radius(pts):
    """Calculate the minimum radius for the given centers."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    # Distance to boundary
    r_norms = np.linalg.norm(pts, axis=1)
    min_boundary_dist = 1.0 - np.max(r_norms)
    
    return min(min_dist / 2.0, min_boundary_dist)

def _local_search(pts, max_iter=2000, kick_prob=0.05, kick_size=0.02):
    """
    Perform local search to maximize the minimum radius.
    Uses a nudge strategy: identify the bottleneck and move particles away.
    """
    pts = pts.copy()
    current_r = _calculate_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    stagnation = 0
    max_stagnation = 100
    
    for i in range(max_iter):
        # Find the bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        r_norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - r_norms
        
        # We want to maximize min(min_dist/2, min_boundary)
        # Let's find which constraint is active.
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        min_boundary = np.min(boundary_dists)
        
        # Identify indices
        # If boundary is the bottleneck
        if min_boundary < min_dist / 2.0:
            # Find circle closest to boundary
            idx = np.argmin(boundary_dists)
            # Move it towards center
            # Direction is -pts[idx]
            move_dir = -pts[idx]
            norm_dir = np.linalg.norm(move_dir)
            if norm_dir > 1e-9:
                move_dir = move_dir / norm_dir
            else:
                move_dir = np.array([1.0, 0.0]) # If at center, move right? (Unlikely to be bottleneck)
            
            # Try a small step
            step = 0.005
            new_pts = pts.copy()
            new_pts[idx] += move_dir * step
            
            new_r = _calculate_radius(new_pts)
            if new_r > current_r:
                pts = new_pts
                current_r = new_r
                stagnation = 0
            else:
                stagnation += 1
                
        else:
            # Pairwise distance is bottleneck
            # Find the pair with min distance
            min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
            i_idx, j_idx = min_idx[0], min_idx[1]
            
            # Move i away from j
            move_dir = pts[i_idx] - pts[j_idx]
            norm_dir = np.linalg.norm(move_dir)
            if norm_dir > 1e-9:
                move_dir = move_dir / norm_dir
            else:
                move_dir = np.array([1.0, 0.0])
            
            step = 0.005
            new_pts = pts.copy()
            new_pts[i_idx] += move_dir * step
            # Also try moving j away from i? Or both?
            # Moving both apart by half step is often better.
            new_pts2 = pts.copy()
            new_pts2[i_idx] += move_dir * (step / 2.0)
            new_pts2[j_idx] -= move_dir * (step / 2.0)
            
            r1 = _calculate_radius(new_pts)
            r2 = _calculate_radius(new_pts2)
            
            if r2 > current_r:
                pts = new_pts2
                current_r = r2
                stagnation = 0
            elif r1 > current_r:
                pts = new_pts
                current_r = r1
                stagnation = 0
            else:
                stagnation += 1
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            
        # Escape local minima
        if stagnation > max_stagnation:
            # Apply a random kick to a random point
            idx = np.random.randint(N)
            pts[idx] += np.random.randn(2) * kick_size
            # Ensure it stays inside the unit circle roughly? 
            # The radius calculation will penalize it if it goes out, but we might get stuck in an invalid state?
            # No, _calculate_radius handles it by taking min with boundary dist.
            # But if pts[idx] goes way out, boundary dist becomes negative?
            # 1 - |c_i| becomes negative. Min will be negative.
            # We should probably project back or just let it be and the "move towards center" logic will fix it.
            # However, if it goes outside, the "bottleneck" detection might be confused.
            # Let's project any point with norm > 0.99 to 0.99 to keep it valid-ish during search.
            r_norms = np.linalg.norm(pts, axis=1)
            mask = r_norms > 0.99
            if np.any(mask):
                pts[mask] = pts[mask] / r_norms[mask, None] * 0.99
                
            stagnation = 0

    return best_pts, best_r

def construct():
    """
    Find the optimal packing of 50 circles in a unit circle.
    Uses multiple restarts of a local search algorithm.
    """
    best_r = 0.0
    best_pts = None
    
    num_restarts = 50
    np.random.seed(42) # For reproducibility
    
    for restart in range(num_restarts):
        # Initialization: Random points inside the circle
        # Generate random points in a square, reject if outside circle
        # Or generate in polar coordinates
        # Polar: r = sqrt(random), theta = random * 2pi
        r = np.sqrt(np.random.rand(N)) * 0.95
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Run local search
        pts_opt, r_opt = _local_search(pts, max_iter=500)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()
            
    # Final refinement with more iterations on the best found
    if best_pts is not None:
        pts_final, r_final = _local_search(best_pts, max_iter=2000, kick_size=0.005)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final

    # Final validity check and projection
    if best_pts is None:
        # Fallback: simple grid
        best_pts = np.zeros((N, 2))
        
    r_norms = np.linalg.norm(best_pts, axis=1)
    mask = r_norms > 0.999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_norms[mask, None] * 0.999

    return best_pts

# EVOLVE-BLOCK-END
