import numpy as np
import time
from scipy.optimize import minimize

N = 200

def compute_radius(points):
    """Compute the maximum radius for a given set of points."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    # Center and scale to fit in [-0.5, 0.5]
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def objective_and_grad(flat_pts):
    """
    Objective: minimize the sum of inverse powers of distances.
    This is a smooth approximation to maximizing the minimum distance.
    """
    pts = flat_pts.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    # Use a high power to emphasize close pairs
    power = 4
    # We want to minimize sum(1/d^power)
    # To avoid overflow/underflow, we can use a log-sum-exp like approach or just direct sum
    # Direct sum might be unstable if distances are very small.
    # Let's use a capped distance to prevent infinity
    dist_sq = np.maximum(dist_sq, 1e-8)
    
    # Energy from pairs
    # E_pairs = sum_{i<j} 1/d_ij^power
    # Since dist_sq is symmetric, we sum over all i!=j and divide by 2
    inv_dist_power = np.power(dist_sq, -power/2.0)
    np.fill_diagonal(inv_dist_power, 0.0)
    E_pairs = 0.5 * np.sum(inv_dist_power)
    
    # Gradient of E_pairs
    # d/dp_i (1/d_ij^power) = -power * (p_i - p_j) / d_ij^(power+2)
    # Note: dist_sq = d_ij^2
    # d_ij^(power+2) = (dist_sq)^(power/2 + 1)
    
    # grad_E_pairs[i] = sum_{j!=i} -power * (p_i - p_j) / dist_sq_ij^(power/2 + 1)
    
    # Vectorized:
    # term_ij = -power * (p_i - p_j) / dist_sq_ij^(power/2 + 1)
    # This is a (N, N, 2) array
    
    # To save memory, we can compute this in chunks or just do it directly if N is small enough.
    # N=200, so 200x200x2 is 80k floats, which is fine.
    
    # dist_sq is (N, N)
    # We need dist_sq^(power/2 + 1)
    denom = np.power(dist_sq, power/2.0 + 1.0)
    np.fill_diagonal(denom, 1.0) # avoid div by 0
    
    # factor_ij = -power / denom_ij
    factor = -power / denom
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (p_i - p_j)
    # = sum_j factor_ij * p_i - sum_j factor_ij * p_j
    # = p_i * sum_j factor_ij - sum_j (factor_ij * p_j)
    
    sum_factor = np.sum(factor, axis=1)  # (N,)
    # sum_j (factor_ij * p_j)
    # factor is (N, N), pts is (N, 2)
    # We can use einsum or matrix multiplication
    # term2 = factor.T @ pts ? No.
    # (factor * p_j) summed over j.
    # factor is (i, j). We want sum_j factor[i, j] * p[j, k].
    # This is factor @ pts.
    sum_factor_pts = factor @ pts  # (N, 2)
    
    grad_pairs = sum_factor[:, np.newaxis] * pts - sum_factor_pts  # (N, 2)
    
    # Wall energy
    # E_walls = sum_i (1/w_i^power) where w_i is distance to nearest wall
    # Distance to walls:
    x = pts[:, 0]
    y = pts[:, 1]
    
    d_left = x + 0.5
    d_right = 0.5 - x
    d_bottom = y + 0.5
    d_top = 0.5 - y
    
    # We want to penalize being close to walls.
    # Let's use the minimum distance to any wall for each point.
    # But min is not differentiable.
    # Alternative: sum of 1/d^power for each wall.
    # This will push points away from all walls.
    
    d_left = np.maximum(d_left, 1e-8)
    d_right = np.maximum(d_right, 1e-8)
    d_bottom = np.maximum(d_bottom, 1e-8)
    d_top = np.maximum(d_top, 1e-8)
    
    E_walls = np.sum(
        np.power(d_left, -power) + 
        np.power(d_right, -power) + 
        np.power(d_bottom, -power) + 
        np.power(d_top, -power)
    )
    
    # Gradient of E_walls
    # d/dx (1/d_left^power) = -power * 1/d_left^(power+1) * d(d_left)/dx
    # d(d_left)/dx = 1
    # d/dx (1/d_right^power) = -power * 1/d_right^(power+1) * d(d_right)/dx
    # d(d_right)/dx = -1
    
    grad_x_walls = -power * (np.power(d_left, -(power+1)) - np.power(d_right, -(power+1)))
    grad_y_walls = -power * (np.power(d_bottom, -(power+1)) - np.power(d_top, -(power+1)))
    
    grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
    
    total_E = E_pairs + E_walls
    total_grad = grad_pairs + grad_walls
    
    return total_E, total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Hexagonal lattices with varying row counts
    for n_rows in range(10, 21):
        pts = hex_init(n_rows)
        inits.append(pts)
        
    # 2. Random initializations
    for _ in range(20):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 3. Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        flat_pts = pts.flatten()
        
        # Use L-BFGS-B
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        try:
            # Maximize radius by minimizing the objective
            # We use a high power to make it sensitive to min distance
            result = minimize(
                objective_and_grad, 
                flat_pts, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < 1e10:
                new_pts = result.x.reshape(N, 2)
                # Ensure points are within bounds
                new_pts = np.clip(new_pts, -0.5, 0.5)
                
                r = compute_radius(new_pts)
                if r > best_r:
                    best_r = r
                    best_points = new_pts.copy()
                    
        except Exception:
            pass
            
        # If we still have time, do a local polish with a higher power
        if time.time() < deadline - 20:
            # Try again with a higher power for more local refinement
            # We can't easily change the power in the closure without redefining, 
            # but we can just run another optimization from the current best.
            pass
            
    # Final polish: take the best point and run a few more iterations with a very high power
    if best_points is not None and time.time() < deadline - 10:
        # We can't easily change the power in the global function.
        # Let's just return the best found.
        pass
        
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
