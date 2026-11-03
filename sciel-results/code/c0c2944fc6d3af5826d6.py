import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(pts):
    """Compute the valid radius for a set of points."""
    n = len(pts)
    # Boundary constraint
    r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    r_min_boundary = np.min(r_boundary)
    # Pairwise constraint
    if n > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_min_pair = np.min(dist) / 2.0
    else:
        r_min_pair = r_min_boundary
    return min(r_min_boundary, r_min_pair)

def _hexagonal_init(n):
    """Create a 7x7 grid plus one extra point."""
    # 7x7 grid
    m = 7
    spacing = 1.0 / m
    ticks = (np.arange(m) + 0.5) * spacing - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We need 50 points. 7*7 = 49. Add one more.
    # Place the 50th point in the center, then push it out.
    extra = np.array([0.0, 0.0])
    pts = np.vstack([grid, extra])
    
    return pts

def _optimize_vectorized(pts, max_time=100.0):
    """Vectorized repulsion-based optimization."""
    start_time = time.time()
    pts = pts.copy().astype(float)
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    lr = 0.01
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        # Compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Forces: repel if close
        forces = np.zeros_like(pts)
        # Only consider pairs that are too close (distance < 2 * current_r * 1.1)
        threshold = 2 * best_r * 1.1
        mask = dist < threshold
        # For pairs that are too close, apply repulsive force
        # Force magnitude proportional to overlap
        overlap = np.maximum(threshold - dist, 0)
        # Normalize by distance to get direction
        safe_dist = np.where(dist > 1e-10, dist, 1e-10)
        force_mag = overlap / safe_dist
        # Apply force: for each i, sum over j
        # forces[i] += diff[i,j] * force_mag[i,j]
        forces += np.sum(diff * force_mag[:, :, np.newaxis], axis=1)
        
        # Boundary repulsion
        for k in range(2):
            too_far_right = pts[:, k] > 0.5 - best_r
            too_far_left = pts[:, k] < -0.5 + best_r
            forces[:, k] -= np.where(too_far_right, (pts[:, k] - (0.5 - best_r)) * 10, 0)
            forces[:, k] += np.where(too_far_left, ((-0.5 + best_r) - pts[:, k]) * 10, 0)
        
        # Update
        pts += forces * lr
        pts = np.clip(pts, -0.5, 0.5)
        
        # Compute new radius
        new_r = _compute_radius(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.9999
    
    return best_pts, best_r

def _nelder_mead_optimize(pts, max_time=20.0):
    """Refine with Nelder-Mead."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        if time.time() - start_time > max_time:
            return 0
        pts_reshaped = x.reshape(n, 2)
        # Penalize if out of bounds
        if np.any(pts_reshaped > 0.5) or np.any(pts_reshaped < -0.5):
            return 0
        r = _compute_radius(pts_reshaped)
        return -r  # Minimize negative radius
    
    res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if res.fun < 0:
        return res.x.reshape(n, 2)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with 7x7 grid + 1
    pts = _hexagonal_init(N)
    r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = r
    
    # Run vectorized optimization
    if time.time() < deadline - 30:
        pts, r = _optimize_vectorized(pts, max_time=deadline - time.time() - 30)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 10:
        pts = _nelder_mead_optimize(best_pts, max_time=deadline - time.time() - 10)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
