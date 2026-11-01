import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 50

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    if np.any(np.abs(p) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _pair_dist_matrix(p):
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    return np.linalg.norm(diffs, axis=2)

def _softmin_radius(p, beta=50.0):
    """Smooth approximation of min radius using softmin."""
    # Boundary
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    # Pairwise
    dists = _pair_dist_matrix(p)
    np.fill_diagonal(dists, np.inf)
    pd = dists / 2.0
    
    # Softmin over all constraints
    # We want to maximize the minimum of {b_i, pd_ij}
    # min(x) = - max(-x)
    # softmax(x) = log(sum(exp(x)))
    # softmin(x) = -softmax(-x)
    
    # To avoid overflow, we shift by the min value
    all_constraints = np.concatenate([b, pd[np.triu_indices_from(pd, k=1)]])
    
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    # exp(-beta * shifted) is safe because shifted >= 0, so -beta*shifted <= 0
    # sum(exp(...)) is between 1 and len(all_constraints)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    softmin = min_val - log_sum / beta
    
    return softmin

def _smooth_obj(p, beta=50.0):
    """Objective to maximize: negative softmin radius."""
    return -_softmin_radius(p, beta)

def _lloyd_step(p, n_steps=1, time_limit=None, start_time=None):
    """
    Perform Lloyd's algorithm steps to equalize Voronoi cell sizes.
    This pushes points away from crowded areas and towards empty areas.
    """
    p = p.copy()
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            vor = Voronoi(p)
        except QhullError:
            # If Voronoi fails (e.g., degenerate points), skip
            break
            
        new_p = p.copy()
        
        for i in range(N):
            # Get the vertices of the Voronoi cell for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded region (point is on the convex hull)
                # Heuristic: Move slightly towards the center of the box
                # This prevents points from flying out to infinity
                center = np.array([0.0, 0.0, 0.0])
                # Blend with current position
                new_p[i] = 0.9 * p[i] + 0.1 * center
            else:
                # Bounded region: Move to the centroid of the vertices
                vertices = vor.vertices[region]
                centroid = np.mean(vertices, axis=0)
                # Move to the centroid
                new_p[i] = centroid
        
        # Clamp to cube with a small margin to avoid exact boundary issues
        # The margin helps with numerical stability
        margin = 1e-6
        new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)
        
        # Optional: Add a small repulsion from walls to prevent sticking
        # If a point is very close to the wall, push it in
        dist_to_wall = 0.5 - np.max(np.abs(new_p), axis=1)
        close_to_wall = dist_to_wall < 0.01
        if np.any(close_to_wall):
            # Push towards center
            center = np.array([0.0, 0.0, 0.0])
            new_p[close_to_wall] += 0.05 * (center - new_p[close_to_wall])
            new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)

        p = new_p
        
    return p

def _repulsion_anneal(p0, target_r, n_steps=200, time_limit=None, start_time=None):
    """Repulsion-based local search to push points apart while respecting target radius."""
    p = p0.copy()
    limits = 0.5 - target_r
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = np.zeros_like(p)
        
        # Boundary forces
        for axis in range(3):
            over_pos = p[:, axis] > limits
            over_neg = p[:, axis] < -limits
            forces[over_pos, axis] -= 1.0
            forces[over_neg, axis] += 1.0
        
        # Pairwise repulsion
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        mask = dists < 2 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            unit_vecs = unit_vecs * mask[:, :, np.newaxis]
            pair_forces = np.sum(unit_vecs, axis=1)
            forces += pair_forces
        
        # Adaptive step size
        step_size = 0.02 * (1.0 - 0.5 * step / n_steps)
        p += step_size * forces
        
        # Clamp to cube
        p = np.clip(p, -0.5, 0.5)
    
    return p

def _seed_fcc():
    """Generate FCC lattice points and select the most compact cluster."""
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]].copy()

def _seed_grid():
    """Generate a 3D grid of points."""
    # 4x4x4 = 64 points, take 50
    coords = np.linspace(-0.4, 0.4, 4)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    grid -= np.mean(grid, axis=0)
    
    dists = np.linalg.norm(grid, axis=1)
    idx = np.argsort(dists)
    
    return grid[idx[:N]].copy()

def _seed_random(seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(-0.4, 0.4, size=(N, 3))
    return p

def _local_search(p0, time_limit, start_time):
    """
    Combined local search:
    1. Lloyd steps for global equalization.
    2. Multi-scale soft-min gradient polishing.
    3. Repulsion annealing if stuck.
    """
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    # Multi-scale beta values for soft-min
    betas = [10.0, 30.0, 100.0, 300.0]
    
    # Main loop
    while time.time() - start_time < time_limit:
        # 1. Lloyd Step
        p_lloyd = _lloyd_step(p, n_steps=2, time_limit=time_limit, start_time=start_time)
        r_lloyd = _compute_radius(p_lloyd)
        
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_p = p_lloyd.copy()
            p = p_lloyd
        else:
            # If Lloyd didn't improve, try polishing the current best
            p = best_p.copy()
            
        # 2. Soft-min Polish
        improved_in_polish = False
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
            
            def obj(x):
                return _smooth_obj(x.reshape(N, 3), beta=beta)
            
            # Use L-BFGS-B
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                # Limit iterations to avoid hanging
                max_iter = 50
                result = minimize(obj, p.flatten(), method='L-BFGS-B', 
                                 bounds=bounds, options={'maxiter': max_iter, 'gtol': 1e-6})
                p_smooth = result.x.reshape(N, 3)
                r_smooth = _compute_radius(p_smooth)
                
                if r_smooth > best_r + 1e-9:
                    best_r = r_smooth
                    best_p = p_smooth.copy()
                    p = p_smooth
                    improved_in_polish = True
            except Exception:
                pass
        
        # 3. If no improvement in a full cycle, try repulsion anneal
        if not improved_in_polish:
            # Try to push apart using current radius
            p_rep = _repulsion_anneal(best_p, best_r, n_steps=100, 
                                      time_limit=time_limit - (time.time() - start_time),
                                      start_time=start_time)
            r_rep = _compute_radius(p_rep)
            if r_rep > best_r + 1e-9:
                best_r = r_rep
                best_p = p_rep.copy()
                p = p_rep
            else:
                # If stuck, break to save time or try a random perturbation
                # For now, break if we've spent too much time without gain
                if time.time() - start_time > time_limit * 0.8:
                    break
                # Small random perturbation
                p = best_p + np.random.normal(0, 0.001, size=(N, 3))
                p = np.clip(p, -0.5, 0.5)
    
    return best_p, best_r

def construct():
    start_time = time.time()
    total_time_limit = 135.0  # Safety margin from 150s
    
    # Try multiple seeds and pick the best
    seeds = [
        ('fcc', _seed_fcc),
        ('grid', _seed_grid),
        ('random', _seed_random),
    ]
    
    best_p = None
    best_r = -1
    
    # Allocate time per seed
    # Give more time to the first (usually best) seed
    time_allocations = [0.5, 0.3, 0.2] # Fractions of available time
    available_time = total_time_limit - 5.0
    
    for i, (seed_name, seed_func) in enumerate(seeds):
        if time.time() - start_time > total_time_limit - 5.0:
            break
        
        p0 = seed_func()
        
        # Scale to fit in cube with some margin
        max_coord = np.max(np.abs(p0))
        if max_coord > 0:
            scale = 0.42 / max_coord
            p0 = p0 * scale
        
        # Local search
        seed_time_limit = time_allocations[i] * available_time
        seed_time_limit = min(seed_time_limit, total_time_limit - 5.0 - (time.time() - start_time))
        if seed_time_limit < 5.0:
            break
        
        p_improved, r_improved = _local_search(p0, seed_time_limit, start_time)
        
        if r_improved > best_r:
            best_r = r_improved
            best_p = p_improved
    
    # Final high-precision polish
    if best_p is not None and time.time() - start_time < total_time_limit - 3.0:
        p = best_p.copy()
        
        # Very high beta for precision
        def obj(x):
            return _smooth_obj(x.reshape(N, 3), beta=1000.0)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(obj, p.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 100, 'gtol': 1e-8})
            p_smooth = result.x.reshape(N, 3)
            r_smooth = _compute_radius(p_smooth)
            
            if r_smooth > best_r:
                best_r = r_smooth
                best_p = p_smooth
        except:
            pass
    
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
    
    return best_p
