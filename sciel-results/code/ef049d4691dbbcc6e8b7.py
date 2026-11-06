"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_radius(centers, beta=100.0):
    """
    Smooth approximation of the radius.
    Uses log-sum-exp to approximate the minimum of distances.
    We want to maximize the minimum distance.
    Let d_ij be distances between pairs, and d_iw be distance to walls.
    We want max(min(d_ij, d_iw)).
    Equivalently, min(-min(d_ij, d_iw)) = min(max(-d_ij, -d_iw)).
    
    A common trick is to minimize the sum of soft repulsions:
    f(x) = (1/beta) * log( sum exp(-beta * d) )
    As beta -> inf, this approaches -min(d).
    Minimizing this function maximizes the minimum distance.
    """
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e-12)
    d = np.sqrt(d2)
    
    # We only need upper triangle to avoid double counting, but summing all is fine
    # d is symmetric.
    
    # Wall distances
    # Distance to each of the 6 walls
    # x, y, z in [-0.5, 0.5]
    # dist to x=-0.5 is x + 0.5
    # dist to x=0.5 is 0.5 - x
    # etc.
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    wall_dists = np.stack([
        x + 0.5, 0.5 - x,
        y + 0.5, 0.5 - y,
        z + 0.5, 0.5 - z
    ], axis=1)
    
    # Combine all distances into one vector
    # Note: d has diagonal 1e-12 (sqrt is ~1e-6), we should ignore diagonal
    d_flat = d.flatten()
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    d_flat = d_flat[mask]
    
    w_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([d_flat, w_flat])
    
    # Avoid log(0) or negative values (shouldn't happen if inside, but safe)
    all_dists = np.maximum(all_dists, 1e-9)
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow in exp, subtract max
    max_d = np.max(all_dists)
    val = np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    return -1.0 / beta * (val - beta * max_d)
    # Wait, the formula is:
    # min(d) approx -1/beta * log(sum(exp(-beta * d)))
    # Let's check limits.
    # If d1 is min. exp(-beta d1) is max.
    # log(sum) approx log(exp(-beta d1)) = -beta d1.
    # -1/beta * (-beta d1) = d1.
    # So minimizing this function minimizes the soft-min distance.
    # We want to MAXIMIZE the radius.
    # So we should MINIMIZE -soft_min_radius.
    
    return -(-1.0 / beta * (val - beta * max_d)) # This returns soft_min_radius. 
    # Actually, let's just return the value to be minimized.
    # Objective: minimize f(x).
    # We want max min_dist.
    # So minimize -min_dist.
    # soft_min_dist approx min_dist.
    # So minimize -soft_min_dist.
    
    # The code above returns -soft_min_dist?
    # Let's re-evaluate.
    # val = log(sum(exp(-beta * (d - max_d))))
    # soft_min = -1/beta * (val - beta * max_d) = -val/beta + max_d
    # This approximates min(d).
    # We want to maximize min(d).
    # So we want to minimize -min(d).
    # So objective should return -soft_min.
    
    # Let's rewrite clearly.
    # return -( -1.0 / beta * (val - beta * max_d) )
    # which is 1.0 / beta * (val - beta * max_d)

def _objective(x):
    centers = x.reshape(N, 3)
    # Soft min radius approximation
    # We want to maximize the minimum distance.
    # Minimize - (soft min distance)
    
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e-12)
    d = np.sqrt(d2)
    
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    wall_dists = np.stack([
        x + 0.5, 0.5 - x,
        y + 0.5, 0.5 - y,
        z + 0.5, 0.5 - z
    ], axis=1)
    
    mask = ~np.eye(n, dtype=bool)
    d_flat = d.flatten()[mask]
    w_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([d_flat, w_flat])
    all_dists = np.maximum(all_dists, 1e-9)
    
    beta = 50.0 # Tuned for stability
    max_d = np.max(all_dists)
    # log-sum-exp trick
    log_sum_exp = np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    soft_min = -1.0 / beta * log_sum_exp + max_d
    
    return -soft_min

def _push_apart(centers, time_budget):
    """
    Local refinement: iteratively find the closest pair (or wall) and push them apart.
    """
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    
    while time.time() - start < time_budget:
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Find min pair
        min_idx = np.unravel_index(np.argmin(d2), d2.shape)
        min_d = np.sqrt(d2[min_idx])
        
        # Check walls
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_d = wall_dists[min_wall_idx]
        
        # Determine if pair or wall is the bottleneck
        # Radius is min(min_d/2, min_wall_d)
        # We want to increase the bottleneck.
        
        # Simple heuristic: if pair distance is small relative to wall distance, push pair.
        # Otherwise push wall.
        # Or just push the one that limits the radius more.
        
        # Let's just do a few steps of pushing the closest pair and the closest wall.
        
        # Push pair
        i, j = min_idx
        if i != j:
            dir_vec = centers[i] - centers[j]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-9:
                dir_vec /= norm
                # Push apart by a small amount
                step = 0.001
                centers[i] += step * dir_vec
                centers[j] -= step * dir_vec
        
        # Push wall
        c = centers[min_wall_idx]
        # Find which wall is closest
        coords = np.abs(c)
        wall_axis = np.argmax(coords)
        sign = np.sign(c[wall_axis])
        # Move away from wall
        step = 0.001
        c[wall_axis] -= sign * step
        centers[min_wall_idx] = c
        
        # Clip to valid range to prevent escape
        centers = np.clip(centers, -0.499, 0.499)
        
        # Early exit if no improvement or time is up
        # For now, just run until time budget.

    return centers

def _bcc_init():
    """
    Initialize centers on a perturbed BCC lattice.
    """
    # BCC lattice points in a unit cell: (0,0,0) and (0.5, 0.5, 0.5)
    # We need 50 points.
    # A 4x4x4 simple grid has 64 points.
    # A 3x3x3 BCC supercell has 3*3*3*2 = 54 points. This is close to 50.
    # Let's use a 3x3x3 BCC supercell and remove 4 points (e.g., corners or random).
    
    # Generate 3x3x3 BCC points
    # Lattice constant a. Let's set a=1/3 to fit in [-0.5, 0.5] roughly?
    # If a=1/3, range is 0 to 1. Center at 0.5.
    # Points:
    # (i, j, k) for i,j,k in 0..2
    # (i+0.5, j+0.5, k+0.5) for i,j,k in 0..2
    
    points = []
    a = 1.0 / 3.0
    offset = -0.5 + a/2 # Center the grid
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner of cell
                x = offset + i * a
                y = offset + j * a
                z = offset + k * a
                points.append([x, y, z])
                
                # Body center of cell
                x_c = offset + (i + 0.5) * a
                y_c = offset + (j + 0.5) * a
                z_c = offset + (k + 0.5) * a
                points.append([x_c, y_c, z_c])
                
    points = np.array(points)
    # We have 54 points. We need 50.
    # Remove 4 points. Let's remove the corners of the supercell to keep it compact?
    # Or just take the first 50.
    # Taking first 50 might be biased. Let's shuffle and take 50.
    np.random.seed(42)
    idx = np.random.permutation(len(points))
    points = points[idx[:N]]
    
    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Initialize with BCC
    centers = _bcc_init()
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    
    # 2. L-BFGS-B Optimization
    # Flatten centers
    x0 = best_centers.flatten()
    
    # Define bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        # Use L-BFGS-B
        # maxiter is high, but we stop if time is up? scipy doesn't support time limit directly.
        # We can wrap the objective to check time, but that's messy.
        # Instead, we run minimize with a reasonable maxiter.
        # L-BFGS-B is fast. 1000 iterations should be fast.
        
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        if res.success or res.fun < _objective(x0):
            new_centers = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                new_r = _radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
    except Exception:
        pass
    
    # 3. Push-apart refinement
    if time.time() - start_time < deadline:
        budget = deadline - (time.time() - start_time) - 5.0
        if budget > 0:
            refined = _push_apart(best_centers, budget)
            refined_r = _radius(refined)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined
    
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        # Fallback to simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers

# EVOLVE-BLOCK-END
