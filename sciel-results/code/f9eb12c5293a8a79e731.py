"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the minimum radius for a set of points."""
    if points is None or len(points) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # If any point is outside, the radius is effectively negative or invalid for packing
    if min_boundary < -1e-9:
        return 0.0

    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_seed(n, jitter=0.0):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius, kind="stable")[:n]
            result = points[idx].copy()
            if jitter > 0:
                result += np.random.normal(0, jitter, size=result.shape)
                # Ensure still inside
                norms = np.linalg.norm(result, axis=1)
                outside = norms > 0.99
                if np.any(outside):
                    result[outside] *= 0.99 / norms[outside, np.newaxis]
            return result
        spacing *= 0.98

def _ring_seed(n, scale=0.18):
    """Generate a ring-based seed for n circles."""
    centers = []
    if n > 0:
        centers.append([0.0, 0.0])
    remaining = n - 1
    
    ring = 1
    while remaining > 0:
        # Number of circles on this ring
        # Circumference ~ 2*pi*r, spacing ~ 2*r_circle (approx)
        # Heuristic: fit as many as possible without obvious overlap
        r = ring * scale
        if r > 0.95: break
        
        # Max circles: 2*pi*r / (2*r) = pi. But r is container radius, not circle radius.
        # Let's just use a fixed capacity estimate that grows with r
        # At r=0.18, 6 circles is tight. 
        # Let's try: count = int(2 * np.pi * r / 0.25)
        max_c = int(2 * np.pi * r / 0.25)
        if max_c < 1: max_c = 1
        
        count = min(remaining, max_c)
        if count == 0:
            ring += 1
            continue
            
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        centers.extend(ring_pts)
        remaining -= count
        ring += 1
        
    pts = np.array(centers)
    if len(pts) < n:
        # Fill remaining with random
        extra_angles = np.random.uniform(0, 2 * np.pi, n - len(pts))
        extra_radii = np.random.uniform(0.5, 0.9, n - len(pts))
        extra_pts = np.column_stack([extra_radii * np.cos(extra_angles), extra_radii * np.sin(extra_angles)])
        pts = np.vstack([pts, extra_pts])
    elif len(pts) > n:
        pts = pts[:n]
        
    # Ensure inside
    norms = np.linalg.norm(pts, axis=1)
    outside = norms > 0.99
    if np.any(outside):
        pts[outside] *= 0.99 / norms[outside, np.newaxis]
    
    return pts

def _random_seed(n):
    """Generate a random seed for n circles."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 0.95, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _energy_and_grad(x):
    """
    Compute energy and gradient for the repulsion model.
    E = sum_{i<j} 1/d_ij^2 + lambda * sum_i max(0, |c_i| - 1)^2
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dists_sq_safe = np.maximum(dists_sq, 1e-12)
    
    # Energy term: sum 1/d^2
    # We sum over i < j
    iu = np.triu_indices(N, k=1)
    e_pair = np.sum(1.0 / dists_sq_safe[iu])
    
    # Gradient of pair term
    # grad_i = sum_{j != i} -2 * (x_i - x_j) / d_ij^4
    mask = ~np.eye(N, dtype=bool)
    factor = np.zeros((N, N))
    factor[mask] = -2.0 / (dists_sq_safe[mask] ** 2)
    
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary penalty
    lambda_bound = 100.0
    norms = np.linalg.norm(pts, axis=1)
    outside = norms > 1.0
    e_bound = 0.0
    grad_bound = np.zeros_like(pts)
    
    if np.any(outside):
        out_norms = norms[outside]
        e_bound = lambda_bound * np.sum((out_norms - 1.0)**2)
        grad_vals = 2.0 * lambda_bound * (out_norms - 1.0) / out_norms
        grad_bound[outside] = grad_vals[:, np.newaxis] * pts[outside]
        
    total_e = e_pair + e_bound
    total_grad = grad_pair + grad_bound
    
    return total_e, total_grad.flatten()

def _optimize_from_seed(seed, deadline):
    """Optimize from a seed using L-BFGS-B."""
    x0 = seed.flatten()
    
    # Check time
    if time.time() > deadline:
        return seed, _score(seed)
        
    try:
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.status == 0:
            pts = res.x.reshape(N, 2)
            # Clip to ensure validity if L-BFGS-B drifted slightly
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                pts[norms > 1.0] *= 0.9999 / norms[norms > 1.0, np.newaxis]
            return pts, _score(pts)
        else:
            return seed, _score(seed)
            
    except Exception:
        return seed, _score(seed)

def _local_search(points, deadline, step_size=0.05, max_iters=10000):
    """
    Perform a local search with hill climbing.
    """
    pts = points.copy()
    current_r = _score(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    # Precompute distance matrix for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    
    norms = np.linalg.norm(pts, axis=1)
    bound_dists = 1.0 - norms
    min_bound = np.min(bound_dists)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Randomly pick a circle to move
        i = np.random.randint(N)
        
        # Try a few random moves
        found_improvement = False
        for _ in range(5):
            delta = np.random.normal(0, step_size, size=2)
            new_c = pts[i] + delta
            r_new_c = np.linalg.norm(new_c)
            if r_new_c > 1.0:
                continue
            
            # Update distance matrix row/col i
            old_row = dist_mat[i].copy()
            new_dists = np.linalg.norm(pts - new_c, axis=1)
            new_dists[i] = np.inf
            dist_mat[i] = new_dists
            dist_mat[:, i] = new_dists
            
            # Update boundary distance
            old_bound = bound_dists[i]
            bound_dists[i] = 1.0 - r_new_c
            
            # Calculate new min radius
            min_pair = np.min(dist_mat) / 2.0
            min_bound = np.min(bound_dists)
            candidate_r = min(min_pair, min_bound)
            
            if candidate_r > current_r:
                # Accept
                pts[i] = new_c
                current_r = candidate_r
                
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
                
                found_improvement = True
                # Slightly increase step size if improving
                step_size *= 1.01
            else:
                # Revert
                dist_mat[i] = old_row
                dist_mat[:, i] = old_row
                bound_dists[i] = old_bound
                
        if not found_improvement:
            # Reduce step size if no improvement
            step_size *= 0.95
            if step_size < 0.001:
                step_size = 0.05 # Reset to allow exploration again, or break if we want strict convergence
                # For now, just reset to continue searching
        else:
            # Cap step size
            if step_size > 0.1:
                step_size = 0.1

    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = -1.0
    
    seeds = []
    
    # 1. Hexagonal
    try:
        seeds.append(_hexagonal_seed(N))
        seeds.append(_hexagonal_seed(N, jitter=0.01))
    except:
        pass
        
    # 2. Random seeds
    for _ in range(5):
        seeds.append(_random_seed(N))
        
    # 3. Ring seeds
    for _ in range(5):
        scale = np.random.uniform(0.15, 0.25)
        seeds.append(_ring_seed(N, scale=scale))
        
    # Shuffle seeds to ensure diversity in time allocation if we run out
    np.random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # First pass: Gradient optimization
        pts, score = _optimize_from_seed(seed, deadline)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
        # Second pass: Local search from the optimized result
        if time.time() < deadline - 5.0: # Leave some time for final check
            pts_ls, score_ls = _local_search(pts, deadline)
            if score_ls > best_score:
                best_score = score_ls
                best_points = pts_ls

    if best_points is None:
        # Fallback
        best_points = _hexagonal_seed(N)
        
    return best_points
# EVOLVE-BLOCK-END
