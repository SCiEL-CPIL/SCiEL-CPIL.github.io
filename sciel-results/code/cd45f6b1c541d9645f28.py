"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 200 equal non-overlapping circles in [-0.5, 0.5]^2 to maximize radius.
    Uses a grid-based initialization followed by gradient ascent on the min-distance objective.
    """
    deadline = time.time() + 110  # safety margin under 120s

    # --- Initialization: square grid ---
    m = int(np.ceil(np.sqrt(N)))  # 14 for N=200
    # We need 14x14=196 points, so we add 4 more. Use a 14x15 grid? No, 14x14=196.
    # Let's use a 14x14 grid and then add 4 points or use a slightly different grid.
    # Actually, a 14x14 grid gives 196 points. We need 200.
    # Let's use a 15x14 grid? 15*14=210. Too many.
    # Let's stick to 14x14 and add 4 extra points in the center or adjust.
    # Better: Use a 14x14 grid, which is 196 points. Then we have 4 extra slots.
    # Alternatively, use a grid that fits 200 better.
    # sqrt(200) ~ 14.14. So 14x14 is close.
    # Let's generate a 14x14 grid and then add 4 points at the center of the last row?
    # Or just use 200 points from a 14x14 grid plus 4 random?
    # Let's use a 14x14 grid and add 4 points at the center of the square, slightly perturbed.
    
    # Actually, a simpler robust init: 
    # Use a 14x14 grid (196 points) and then add 4 points in the middle of the grid cells.
    # Or just use a 200-point grid by taking the first 200 from a larger grid?
    # Let's use a 14x14 grid and then add 4 points at (0,0) with small offsets.
    
    # Better approach: Use a 14x14 grid, which is 196 points.
    # Then add 4 points at the center of the 4 central cells?
    # Let's just use a 14x14 grid and then add 4 points at the center of the square.
    
    # Let's try a 14x14 grid first.
    m = 14
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We have 196 points. Need 4 more.
    # Add 4 points at the center of the square, slightly offset.
    center = np.array([0.0, 0.0])
    extra = np.array([
        [0.01, 0.01],
        [-0.01, 0.01],
        [0.01, -0.01],
        [-0.01, -0.01]
    ])
    pts = np.vstack([grid, extra])
    
    # Shuffle to avoid symmetry issues
    rng = np.random.default_rng(42)
    pts = pts[rng.permutation(N)]
    
    # --- Objective function ---
    def compute_radius(pts):
        # Compute minimum distance between any two points
        # and minimum distance to boundary
        # dist = min_{i<j} |c_i - c_j|
        # boundary_dist = min_i min_k (0.5 - |c_ik|)
        # radius = min(dist, boundary_dist) / 2
        
        # Efficient computation using scipy.spatial.distance
        from scipy.spatial.distance import pdist
        dists = pdist(pts)
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        
        # Boundary distance
        # For each point, min distance to any of the 4 boundaries
        # boundary_dist_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # But since pts are in [-0.5, 0.5], 0.5 - |x_i| and 0.5 - |y_i|
        abs_pts = np.abs(pts)
        boundary_dists = np.minimum(0.5 - abs_pts[:, 0], 0.5 - abs_pts[:, 1])
        min_boundary = np.min(boundary_dists)
        
        return min(min_dist, min_boundary) / 2.0
    
    def compute_radius_and_grad(pts):
        # Compute radius and gradient for gradient ascent
        # Use a soft-min approximation for differentiability
        # Or use subgradient
        from scipy.spatial.distance import pdist
        
        # Compute all pairwise distances
        dists = pdist(pts)
        # Also boundary distances
        abs_pts = np.abs(pts)
        boundary_dists = np.minimum(0.5 - abs_pts[:, 0], 0.5 - abs_pts[:, 1])
        
        # We want to maximize min(min_dist, min_boundary)
        # Let's use a soft-min: -log(sum(exp(-x_i / t))) * t
        # But this is complex. Let's use a simpler approach:
        # Gradient ascent on the sum of squared distances, but weighted by proximity to min.
        
        # For now, use a simple subgradient method:
        # Find the pair (i,j) with minimum distance, and the point with minimum boundary distance.
        # Move them apart.
        
        min_dist = np.min(dists)
        min_boundary = np.min(boundary_dists)
        
        radius = min(min_dist, min_boundary) / 2.0
        
        # Gradient: 
        # If min_dist < min_boundary, we want to increase the minimum pairwise distance.
        # The subgradient for the minimum is the unit vector in the direction that increases the minimum.
        # For pairwise distance d_ij, the gradient w.r.t. c_i is (c_i - c_j)/d_ij, and w.r.t. c_j is (c_j - c_i)/d_ij.
        # For boundary distance, if a point is closest to the right boundary, gradient w.r.t. c_i is (-1, 0).
        
        grad = np.zeros_like(pts)
        
        if min_dist < min_boundary:
            # Find the pair with minimum distance
            min_idx = np.argmin(dists)
            # Convert pdist index to (i, j)
            # pdist returns distances in order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
            # We need to map min_idx to (i, j)
            i = 0
            j = 1
            idx = 0
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if idx == min_idx:
                        break
                    idx += 1
                if idx == min_idx:
                    break
            
            d = dists[min_idx]
            if d > 1e-12:
                diff = pts[i] - pts[j]
                grad[i] += diff / d
                grad[j] -= diff / d
        else:
            # Find the point with minimum boundary distance
            min_boundary_idx = np.argmin(boundary_dists)
            pt = pts[min_boundary_idx]
            # Determine which boundary is closest
            if pt[0] > 0:
                grad[min_boundary_idx, 0] -= 1.0
            else:
                grad[min_boundary_idx, 0] += 1.0
            if pt[1] > 0:
                grad[min_boundary_idx, 1] -= 1.0
            else:
                grad[min_boundary_idx, 1] += 1.0
        
        return radius, grad
    
    # --- Optimization ---
    # Use gradient ascent with adaptive step size
    lr = 0.01
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Use a more robust optimization: scipy.optimize.minimize with a smooth objective
    # Let's try using scipy.optimize.minimize on the negative of a smooth approximation of the min-distance.
    
    # Smooth approximation: -t * log(sum(exp(-d_ij / t))) for pairwise distances
    # This is expensive for N=200 (19900 pairs).
    
    # Alternative: Use a simple iterative method:
    # For each point, move it in the direction away from its nearest neighbor and boundary.
    
    # Let's use a simple local search:
    # For each point, try small perturbations and keep the best.
    
    # Actually, let's use scipy.optimize.minimize with a BFGS method on a smooth objective.
    # Objective: maximize min(min_dist, min_boundary)
    # Equivalent to: minimize -min(min_dist, min_boundary)
    # This is non-smooth. Use a smooth approximation.
    
    # Let's use a different approach: 
    # 1. Start with the grid.
    # 2. Use a simple gradient ascent with a small step size.
    # 3. Periodically check the time.
    
    # Let's implement a simple gradient ascent:
    pts = best_pts.copy()
    lr = 0.005
    
    # Precompute neighbor list for efficiency? 
    # For N=200, pairwise distances are 19900, which is manageable.
    
    # Use a more efficient gradient computation:
    # For each point, the gradient of the min-distance is the sum of unit vectors away from all points,
    # weighted by how close they are to the minimum.
    
    # Let's use a soft-min gradient:
    # f(pts) = -t * log(sum_{i<j} exp(-d_ij / t) + sum_i exp(-b_i / t))
    # where d_ij are pairwise distances and b_i are boundary distances.
    # Gradient: 
    # df/dc_k = (1/t) * [sum_{j!=k} exp(-d_kj / t) * (c_k - c_j)/d_kj + exp(-b_k / t) * (boundary_grad_k)] / sum(...)
    
    # This is expensive but feasible for N=200.
    
    # Let's implement this:
    t = 0.01  # soft-min parameter
    lr = 0.01
    
    for step in range(1000):
        if time.time() > deadline - 10:
            break
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        dists = pdist(pts)
        
        # Boundary distances
        abs_pts = np.abs(pts)
        boundary_dists = np.minimum(0.5 - abs_pts[:, 0], 0.5 - abs_pts[:, 1])
        
        # Soft-min objective
        exp_dists = np.exp(-dists / t)
        exp_boundaries = np.exp(-boundary_dists / t)
        
        # Sum of exps
        total_exp = np.sum(exp_dists) + np.sum(exp_boundaries)
        
        # Gradient
        grad = np.zeros_like(pts)
        
        # Pairwise contributions
        # For each pair (i, j), the contribution to grad[i] is exp_dists[idx] * (pts[i] - pts[j]) / dists[idx] / t
        # and to grad[j] is exp_dists[idx] * (pts[j] - pts[i]) / dists[idx] / t
        
        # This is O(N^2) which is 19900 for N=200. Manageable.
        
        # We need to map pdist indices to (i, j)
        # This is slow in Python. Let's use a vectorized approach.
        
        # Alternative: Use a simpler gradient:
        # For each point, the gradient is the sum of unit vectors away from all other points,
        # weighted by exp(-d_ij / t).
        
        # Let's compute the gradient for each point:
        for i in range(N):
            # Distance from point i to all others
            diffs = pts[i] - pts
            dists_i = np.sqrt(np.sum(diffs**2, axis=1))
            dists_i[i] = np.inf  # avoid division by zero
            
            # Weights
            weights = np.exp(-dists_i / t)
            weights[i] = 0.0
            
            # Gradient contribution from other points
            # grad[i] += sum_j weights[j] * (pts[i] - pts[j]) / dists_i[j]
            # But we need to be careful with the soft-min formulation.
            
            # Actually, the gradient of -t*log(sum(exp(-d/t))) is:
            # (1/t) * sum_j exp(-d_ij/t) * (c_i - c_j)/d_ij / sum_k exp(-d_ik/t)
            
            # This is getting complex. Let's use a simpler approach:
            # Use a local search with random perturbations.
            pass
        
        # Let's use a different strategy: 
        # Use scipy.optimize.minimize with a BFGS method on a smooth objective.
        # Objective: sum of exp(-d_ij / t) + sum exp(-b_i / t)
        # Minimize this to maximize the minimum distance.
        
        break  # Exit the loop and use scipy.optimize.minimize
    
    # Use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    def objective(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Compute pairwise distances
        dists = pdist(pts)
        # Boundary distances
        abs_pts = np.abs(pts)
        boundary_dists = np.minimum(0.5 - abs_pts[:, 0], 0.5 - abs_pts[:, 1])
        
        # Soft-min objective
        t = 0.01
        return np.sum(np.exp(-dists / t)) + np.sum(np.exp(-boundary_dists / t))
    
    # Initial point
    x0 = pts.flatten()
    
    # Optimize
    result = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-6})
    
    if result.success:
        new_pts = result.x.reshape(N, 2)
        new_radius = compute_radius(new_pts)
        if new_radius > best_radius:
            best_pts = new_pts
            best_radius = new_radius
    
    return best_pts
# EVOLVE-BLOCK-END
