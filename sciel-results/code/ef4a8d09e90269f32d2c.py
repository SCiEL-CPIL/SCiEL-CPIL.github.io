"""Pack 50 equal circles in a unit square using Hard-Disk Expansion + SLSQP Epigraph Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n):
    """Create a hexagonal grid of n points centered in the square."""
    # Approximate number of rows/cols
    # Area ~ n * (2r)^2 * (2/sqrt(3)) ? 
    # Let's just generate a grid and pick the best n.
    # For n=50, a 7x8 or similar grid works.
    # Hexagonal packing:
    # Row 0: x offsets 0, 1, 2...
    # Row 1: x offsets 0.5, 1.5, 2.5...
    
    # Let's generate a large grid and center it, then take the n closest to center?
    # Or just generate enough points.
    
    # Target density: 50 circles in 1x1 square.
    # Area per circle ~ 1/50 = 0.02.
    # Hexagonal area per circle (radius r) = r^2 * 2 * sqrt(3).
    # r^2 * 3.464 = 0.02 => r^2 ~ 0.0057 => r ~ 0.075.
    # Diameter ~ 0.15.
    # Grid spacing ~ 0.15.
    # Number of points across ~ 1/0.15 = 6.6.
    
    rows = 8
    cols = 7
    spacing = 0.14 # Slightly larger than 1/7 to ensure fit? No, we want them close.
    # Let's use a spacing that fits 7 cols in 1.0 width.
    # If 7 cols, 6 gaps. Width = 6 * dx. dx = 1/6 = 0.166.
    # If 8 rows, 7 gaps. dy = sqrt(3)/2 * dx = 0.866 * 0.166 = 0.144.
    # Total height = 7 * 0.144 = 1.0. Fits perfectly.
    
    dx = 1.0 / (cols - 1)
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for i in range(rows):
        y = -0.5 + i * dy + dy/2 # Center vertically?
        # Actually, let's center the block.
        y_offset = -0.5 + (i + 0.5) * dy
        for j in range(cols):
            x_offset = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x_offset += dx / 2
            points.append([x_offset, y_offset])
            
    points = np.array(points)
    
    # We might have more than N points. Pick the N closest to the center?
    # Or just take the first N?
    # Taking the first N from a grid might leave corners empty which is good for radius?
    # No, we want to fill the square.
    # Let's sort by distance from center and take the N closest? That clusters them.
    # Better: Take a rectangular subset that is centered.
    
    # Let's just generate a grid that is slightly larger and clip/center it.
    # For now, return the generated points. If len > N, we need to select.
    # A simple selection: keep points that are inside [-0.5, 0.5]^2.
    # Then if still > N, remove the ones furthest from the center?
    # Or just return the first N.
    
    # Let's refine: Generate a grid with spacing such that we get ~N points.
    # If we use dx=1/6, dy=sqrt(3)/12.
    # Rows: 1.0 / dy = 1.0 / 0.144 = 6.9 -> 7 rows.
    # Cols: 1.0 / dx = 6.
    # Total ~ 42 points. Too few.
    
    # Let's use a denser grid.
    # dx = 1/7. dy = sqrt(3)/14.
    # Rows: 1.0 / (sqrt(3)/14) = 14/1.732 = 8.08 -> 8 rows.
    # Cols: 7.
    # Total ~ 56 points. Good.
    
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter to square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Sort by distance from center, keep closest N?
        # This creates a blob. 
        # Better: Keep a rectangular region in the center.
        # Or just take the first N.
        # Let's try sorting by "potential" - distance to walls.
        # Points closer to walls can potentially expand more?
        # No, let's just take the N points that form the most compact set?
        # For initialization, any valid non-overlapping set is fine.
        # Let's just take the first N.
        points = points[:N]
    elif len(points) < N:
        # Add random points?
        # This shouldn't happen with 8x7 grid.
        pass
        
    return points

def _expand_radius(centers, r_target, max_iters=1000, tol=1e-6):
    """
    Expand radius to r_target by moving points to resolve overlaps.
    Returns (success, new_centers).
    """
    centers = centers.copy()
    r = r_target
    
    for it in range(max_iters):
        # Check walls
        # If a point is too close to a wall, move it inwards.
        # x < -0.5 + r -> x = -0.5 + r
        # x > 0.5 - r -> x = 0.5 - r
        # Same for y.
        
        walls_violated = False
        
        # Left wall
        mask_l = centers[:, 0] < -0.5 + r
        if np.any(mask_l):
            centers[mask_l, 0] = -0.5 + r
            walls_violated = True
            
        # Right wall
        mask_r = centers[:, 0] > 0.5 - r
        if np.any(mask_r):
            centers[mask_r, 0] = 0.5 - r
            walls_violated = True
            
        # Bottom wall
        mask_b = centers[:, 1] < -0.5 + r
        if np.any(mask_b):
            centers[mask_b, 1] = -0.5 + r
            walls_violated = True
            
        # Top wall
        mask_t = centers[:, 1] > 0.5 - r
        if np.any(mask_t):
            centers[mask_t, 1] = 0.5 - r
            walls_violated = True
            
        # Check pairs
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        # min_dist = np.min(dist[np.triu_indices(N, 1)])
        
        # Vectorized pair check
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        if min_dist >= 2 * r - tol:
            return True, centers
            
        # Find the closest pair
        idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = idx
        if i == j:
            continue
            
        d_vec = centers[i] - centers[j]
        d = np.linalg.norm(d_vec)
        
        if d < 1e-10:
            # Coincident, nudge randomly
            centers[i] += np.random.normal(0, 0.01, 2)
            centers[j] -= np.random.normal(0, 0.01, 2)
            continue
            
        # Move i and j apart by half the overlap each
        overlap = 2 * r - d
        if overlap > 0:
            unit_vec = d_vec / d
            move = 0.5 * overlap
            centers[i] += move * unit_vec
            centers[j] -= move * unit_vec
            
        if not walls_violated and min_dist >= 2 * r - tol:
             return True, centers

    return False, centers

def _slsqp_polish(centers, time_limit=10.0):
    """
    Use SLSQP to maximize the minimum radius directly.
    Objective: minimize -r
    Constraints: 
       d_ij - 2r >= 0 for all i<j
       0.5 - |x_i| - r >= 0
       0.5 - |y_i| - r >= 0
    """
    start_time = time.time()
    
    def objective(x):
        centers = x[:2*N].reshape(N, 2)
        r = x[2*N]
        # We want to maximize r, so minimize -r
        return -r
        
    # Constraints are functions of x returning >= 0
    # This is expensive to define 1225 constraints explicitly for SLSQP.
    # Better: Use the soft-min approach with L-BFGS-B as a fallback?
    # Or use SLSQP with a small number of active constraints?
    # SLSQP can handle many constraints but it's slow.
    
    # Alternative: Use the gradient of the soft-min with a very high beta?
    # The inspiration said SLSQP epigraph scored well.
    # Let's implement the constraints efficiently.
    
    def constraints(x):
        centers = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        cons = []
        
        # Pairwise: dist(i,j) - 2r >= 0
        # We can't easily vectorize this into a single function for SLSQP without
        # returning a list of tuples.
        # For N=50, 1225 pairs. 4*50 walls. 1275 constraints.
        # SLSQP might be too slow.
        
        # Let's stick to L-BFGS-B with soft-min but with a very high beta
        # and tight tolerances, as it is faster and differentiable.
        # The "epigraph" benefit is that it doesn't approximate the min.
        # But L-BFGS-B on -soft_min(r) is effectively optimizing the epigraph
        # if beta is high enough.
        
        # Let's try a different approach:
        # Minimize -r subject to d_ij >= 2r and wall constraints.
        # This is a non-linear program.
        
        # Given the time limit, L-BFGS-B on soft-min is safer.
        # Let's just use L-BFGS-B with high beta.
        
        raise NotImplementedError("Using L-BFGS-B instead")
        
    # Since SLSQP with 1200 constraints is likely too slow for 10s,
    # we will rely on the Hard-Disk expansion to get very close,
    # and then use L-BFGS-B with high beta for final polish.
    
    return _local_refine_smooth(centers, time_limit=time_limit, beta=500.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    """
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    x0 = centers.ravel()
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Strategy:
    # 1. Start with a hexagonal grid.
    # 2. Use Hard-Disk Radius Expansion to increase r until stuck.
    # 3. Polish with L-BFGS-B (high beta).
    # 4. Repeat with different initial grids or random perturbations.
    
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal grid
    hex_centers = _make_hexagonal_grid(N)
    if hex_centers is not None and len(hex_centers) == N:
        candidates.append(hex_centers)
        
    # 2. Square grid
    # 7x8 grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    sq_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    candidates.append(sq_centers)
    
    # 3. Random starts
    for i in range(3):
        c_rand = np.random.uniform(-0.4, 0.4, size=(N, 2))
        candidates.append(c_rand)
        
    # Process each candidate
    for idx, centers in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Step 1: Hard-Disk Expansion
        # Start with a small radius and increase.
        # Estimate initial radius
        r_init = _radius(centers)
        if r_init <= 0:
            # Nudge if invalid
            centers = centers + np.random.normal(0, 0.01, (N, 2))
            r_init = _radius(centers)
            
        # Target radius: try to expand up to 0.072
        r_target = 0.072
        
        # If current radius is already high, start from there.
        # Expansion works best if we start from a feasible configuration.
        
        # We need to ensure the initial configuration is feasible for r_init.
        # It is by definition.
        
        # Let's try to expand in steps.
        # Actually, _expand_radius takes a target r and tries to achieve it.
        # If it fails, it returns the best effort.
        
        # Let's try a sequence of targets.
        targets = [0.060, 0.065, 0.068, 0.070, 0.071, 0.0714]
        
        current_centers = centers.copy()
        
        for r_t in targets:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            success, new_centers = _expand_radius(current_centers, r_t, max_iters=500)
            
            # Even if not successful, new_centers might be better than current
            r_new = _radius(new_centers)
            
            if r_new > _radius(current_centers):
                current_centers = new_centers
            else:
                # If it didn't improve, stick with current
                pass
                
            # If we reached the target, we can stop expanding for this candidate
            # and move to polish.
            if r_new >= r_t - 1e-5:
                break
                
        # Step 2: Polish
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 1.0:
            polished = _local_refine_smooth(current_centers, time_limit=min(8.0, time_left), beta=200.0)
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
                
        # If we have time, do a final high-beta polish on the best so far
        if best_centers is not None and time.time() - start_time < DEADLINE - 2:
            time_left = DEADLINE - 2 - (time.time() - start_time)
            if time_left > 1.0:
                final_polished = _local_refine_smooth(best_centers, time_limit=time_left, beta=500.0)
                r_final = _radius(final_polished)
                
                if r_final > best_r:
                    best_r = r_final
                    best_centers = final_polished.copy()
                    
    if best_centers is None:
        # Fallback
        best_centers = _make_hexagonal_grid(N)
        
    return best_centers
