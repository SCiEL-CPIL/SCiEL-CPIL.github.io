"""
Packing 300 equal circles in a unit circle.
Strategy:
1. Generate multiple initial configurations (hexagonal lattice with varying spacing/rotation, random).
2. Use a fast, vectorized local optimization loop (repulsion + boundary constraint).
3. Use scipy.spatial.cKDTree or cdist for efficient nearest-neighbor checks if needed, 
   but for N=300, a simple vectorized distance matrix calculation is often sufficient 
   and avoids overhead, though cdist is better for repeated calls.
   Actually, for the inner loop, we want to update points and check min distance.
   A full NxN matrix calculation is 90k ops, which is very fast in NumPy.
"""

import numpy as np
import time
import random

N = 300

def compute_radius(pts):
    """
    Compute the maximum radius r such that circles of radius r centered at pts
    are non-overlapping and inside the unit circle.
    r = min( min_{i<j} ||p_i - p_j|| / 2, min_i (1 - ||p_i||) )
    """
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Boundary constraint: min distance to boundary
    dists_origin = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(dists_origin)
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distance constraint
    # Use cdist for efficiency and clarity
    # D[i, j] is distance between pts[i] and pts[j]
    # We need min over i != j
    # For N=300, cdist is fast enough.
    try:
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
    except ImportError:
        # Fallback to manual calculation if scipy is missing (unlikely per prompt)
        min_pair_dist = 1.0
        for i in range(N):
            dx = pts[i, 0] - pts[:i, 0]
            dy = pts[i, 1] - pts[:i, 1]
            d = np.hypot(dx, dy)
            if len(d) > 0:
                m = np.min(d)
                if m < min_pair_dist:
                    min_pair_dist = m
    
    return min(min_pair_dist / 2.0, min_boundary)

def generate_hex_initial(spacing, rotation=0.0):
    """Generate a hexagonal lattice configuration."""
    sqrt3 = np.sqrt(3.0)
    # Lattice vectors
    # a1 = (spacing, 0)
    # a2 = (spacing/2, spacing*sqrt(3)/2)
    
    # Determine range of indices
    # Max radius is 1.0.
    # Roughly need indices up to 1/spacing + margin
    k = int(1.0 / spacing) + 3
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            # Hex lattice coordinates
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            
            # Apply rotation
            if rotation != 0.0:
                x_new = x * np.cos(rotation) - y * np.sin(rotation)
                y_new = x * np.sin(rotation) + y * np.cos(rotation)
            else:
                x_new = x
                y_new = y
            
            r = np.hypot(x_new, y_new)
            if r < 1.0:
                points.append((x_new, y_new))
    
    points = np.array(points)
    if len(points) < N:
        return None
    
    # Select the N points closest to the origin
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:N]
    return points[idx]

def generate_random_initial():
    """Generate a random initial configuration inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def optimize_local(pts, max_iters=5000, step_size=0.01, time_limit=10.0):
    """
    Local optimization using a simple gradient-free or force-directed approach.
    We will use a "repulsion" method:
    - If a pair is too close (dist < 2r_target), push them apart.
    - If a point is too close to boundary (dist > 1-r_target), push it inward.
    
    However, defining r_target is tricky. 
    Alternative: Simple local search by perturbing points and accepting if r improves.
    For speed, we can use a "best improvement" strategy or simulated annealing.
    
    Given N=300, a full pairwise check is 45k pairs. 
    In Python, looping over 45k pairs is slow (~5ms per iteration).
    5000 iterations * 5ms = 25 seconds. This is too slow for multiple restarts.
    
    We need a faster inner loop or fewer iterations.
    
    Let's use a vectorized force calculation.
    Force on i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2 * strength
    Plus boundary force if outside 1-r_est.
    
    Actually, a common efficient heuristic for packing is:
    1. Estimate current r.
    2. Identify "bottleneck" pairs (closest pairs) and boundary points.
    3. Move points slightly away from their nearest neighbors and towards center if near boundary.
    
    Let's implement a vectorized gradient step for the objective:
    Maximize r.
    This is equivalent to minimizing -r.
    But r is non-differentiable (min function).
    
    Let's stick to a robust local search with vectorized distance checks.
    To speed up, we can use `scipy.spatial.cKDTree` to find nearest neighbors quickly.
    But for N=300, cdist is likely faster than building a tree every iteration.
    
    Let's try a different approach: 
    Use `scipy.optimize.minimize` with a custom objective?
    Objective: -min(min_pair_dist/2, min_boundary)
    This is non-smooth.
    
    Better: Use a penalty method or just a fast local search.
    
    Let's optimize the local search loop using NumPy vectorization for the distance calculation.
    
    def fast_radius(pts):
        # Vectorized min pair distance
        # This is still O(N^2) memory and compute.
        # For N=300, 300x300 matrix is 90k floats, ~720KB. Very fast.
        D = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
        min_bound = 1.0 - np.max(np.sqrt(np.sum(pts**2, axis=1)))
        return min(min_pair / 2.0, min_bound)
        
    This function is fast in NumPy. 
    Let's estimate: 300^3 ops for the subtraction? No, 300*300*2 for sub, 300*300 for sum.
    It's very fast, likely < 1ms.
    
    So we can do many iterations.
    """
    current_pts = pts.copy()
    current_r = compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = rng.integers(N)
        old_pos = current_pts[idx].copy()
        
        # Propose a new position
        # Use a small random step
        step = rng.uniform(-step_size, step_size, 2)
        new_pos = old_pos + step
        
        # Constrain to circle (soft constraint or hard?)
        # Hard constraint: if outside, project back.
        r_mag = np.hypot(new_pos[0], new_pos[1])
        if r_mag > 0.99: # Keep slightly inside to avoid boundary issues
            new_pos = new_pos / r_mag * 0.99
        
        current_pts[idx] = new_pos
        new_r = compute_radius(current_pts)
        
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Revert
            current_pts[idx] = old_pos
    
    return best_pts, best_r

def construct():
    deadline = time.time() + 140.0
    
    def time_check():
        return time.time() < deadline
    
    best_points = None
    best_radius = 0.0
    
    # 1. Try multiple hexagonal lattice configurations
    # Spacing around optimal ~ 0.106
    # Optimal r ~ 0.053, so spacing 2r ~ 0.106
    spacings = [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112]
    rotations = [0.0, 0.1, 0.2, 0.3, np.pi/6, np.pi/4]
    
    for sp in spacings:
        if not time_check():
            break
        for rot in rotations:
            if not time_check():
                break
            pts = generate_hex_initial(sp, rot)
            if pts is None:
                continue
            
            # Quick evaluation
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_points = pts.copy()
                
                # Run local optimization on this good candidate
                opt_pts, opt_r = optimize_local(pts, max_iters=2000, step_size=0.02, time_limit=5.0)
                if opt_r > best_radius:
                    best_radius = opt_r
                    best_points = opt_pts.copy()
    
    # 2. Try random initializations
    for _ in range(10):
        if not time_check():
            break
        pts = generate_random_initial()
        r = compute_radius(pts)
        # Random starts are usually bad, but sometimes help escape local minima
        # We might skip optimization on random starts if time is tight,
        # but let's try a few.
        if r > 0.04: # Only optimize if reasonably good
            opt_pts, opt_r = optimize_local(pts, max_iters=1000, step_size=0.02, time_limit=5.0)
            if opt_r > best_radius:
                best_radius = opt_r
                best_points = opt_pts.copy()
    
    # 3. Final refinement on the best point found
    if best_points is not None:
        # Do a final, tighter local search
        final_pts, final_r = optimize_local(best_points, max_iters=5000, step_size=0.005, time_limit=10.0)
        if final_r > best_radius:
            best_radius = final_r
            best_points = final_pts.copy()
    
    # Fallback if no points found (should not happen)
    if best_points is None:
        best_points = generate_hex_initial(0.106)
        if best_points is None:
            best_points = generate_random_initial()
    
    # Final validation
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        best_points = generate_random_initial()
    
    # Ensure strictly inside
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = dists_origin >= 1.0 - 1e-10
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-10) / dists_origin[mask]
    
    return best_points

# For testing/verification
if __name__ == "__main__":
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Score (raw): {r}")
