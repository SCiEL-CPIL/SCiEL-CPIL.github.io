"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def compute_radius(coords):
    """
    Computes the maximum radius for the given circle centers.
    Returns the minimum of:
    1. Half the minimum distance between any two circle centers.
    2. The minimum distance from any circle center to the boundary of the square [-0.5, 0.5]^2.
    """
    if not np.all(np.isfinite(coords)):
        return 0.0
    
    # Distance to boundaries
    # x distance to left/right walls
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    # y distance to top/bottom walls
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    
    # Minimum distance to any wall
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    # If any center is outside, the radius is effectively 0 or negative (invalid)
    if min_wall_dist < 0:
        return 0.0

    # Pairwise distances
    # We can compute this efficiently using broadcasting
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    
    # To save memory and time, we can iterate or use a trick.
    # For N=200, N^2 = 40,000, which is small.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Minimum distance between distinct circles
    # Mask the diagonal (distance to self is 0)
    min_pair_dist = np.inf
    for i in range(N):
        dists[i, i] = np.inf
    min_pair_dist = np.min(dists)
    
    radius = min(min_wall_dist, min_pair_dist / 2.0)
    
    return radius

def objective(coords):
    """
    Objective function for minimization. We want to MAXIMIZE the radius,
    so we MINIMIZE the negative radius.
    """
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    """
    Generates an initial hexagonal layout of n points inside the square.
    """
    # Approximate number of rows
    # Area of square is 1. Area of hex cell for radius r is approx r^2 * sqrt(3).
    # Total area n * r^2 * sqrt(3) <= 1 => r <= 1/sqrt(n*sqrt(3))
    # Let's just generate a dense hex grid and crop.
    
    # Side length of the hex "grid" in terms of circle radius r
    # Horizontal spacing = 2r
    # Vertical spacing = r * sqrt(3)
    
    # Let's estimate r from the square grid to start close
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    # Number of points per row in hex packing
    # Width 1.0. Spacing 2r_est.
    # cols = int(1.0 / (2 * r_est)) + 1
    cols = int(1.0 / (2 * r_est)) + 2
    
    # Height 1.0. Spacing r_est * sqrt(3).
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        # Offset alternating rows
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter points inside the square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # We need exactly N points.
    if len(points) < N:
        # This shouldn't happen with the estimation, but just in case, fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Shuffle to randomize selection? Or pick the ones closest to the center?
    # Actually, a contiguous block of hex packing is usually good.
    # Let's pick N points. If we have more, we can just take the first N.
    # However, taking the first N might bias to the top-left.
    # Better: Compute distance from center (0,0) and pick the N closest to form a compact cluster.
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)
    points = points[indices[:N]]
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Initial guess: Hexagonal
    coords = initial_hex_layout(N)
    
    # Initial radius
    best_radius = compute_radius(coords)
    best_coords = coords.copy()
    
    # Optimization
    # Use Nelder-Mead or Powell for derivative-free optimization.
    # Powell is often robust for high dimensions.
    # N=200 -> 400 variables. This is high for standard optimizers.
    # However, the landscape is relatively smooth.
    
    # Let's try a few restarts with different initial conditions if time permits.
    # But first, let's just optimize the hex layout.
    
    # Reshape coords to 1D for scipy
    x0 = coords.flatten()
    
    # Bounds to keep them roughly inside (scipy minimize with method='Powell' doesn't support bounds directly in older versions, 
    # but 'L-BFGS-B' does. However, gradient calculation for min-dist is tricky.
    # Let's use Nelder-Mead or Powell without bounds, and rely on the objective function returning 0 if outside.
    # But if it goes outside, the radius is 0, which is the minimum we want to avoid (we maximize).
    # Actually, if it goes outside, the penalty is huge (radius becomes negative or 0).
    # We want to maximize radius. If radius is 0, the objective is 0.
    # The optimizer will try to find a point with radius > 0.
    
    # A better approach for bounded optimization:
    # Map parameters to the square.
    # But that complicates the gradient.
    
    # Let's stick to a robust local search.
    # We can implement a simple coordinate descent or use scipy.optimize.minimize with method='Powell'.
    
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-10})
        
        # Check time
        if time.time() < deadline:
            new_coords = res.x.reshape(N, 2)
            new_radius = compute_radius(new_coords)
            if new_radius > best_radius:
                best_radius = new_radius
                best_coords = new_coords
    except Exception as e:
        pass
        
    # If we have time, try another optimization from a slightly perturbed start
    if time.time() < deadline - 10:
        # Perturb best_coords
        noise = np.random.normal(0, best_radius * 0.1, (N, 2))
        perturbed = best_coords + noise
        # Ensure inside
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.5, 0.5)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.5, 0.5)
        
        x1 = perturbed.flatten()
        try:
            res1 = minimize(objective, x1, method='Powell', options={'maxiter': 500, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-10})
            if time.time() < deadline:
                new_coords = res1.x.reshape(N, 2)
                new_radius = compute_radius(new_coords)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_coords = new_coords
        except Exception as e:
            pass

    # Final validation
    if not np.all(np.isfinite(best_coords)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_coords

# EVOLVE-BLOCK-END
