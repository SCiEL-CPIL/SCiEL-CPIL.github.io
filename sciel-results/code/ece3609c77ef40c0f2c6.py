import numpy as np
from scipy.spatial import cKDTree
import time

N = 200

def _compute_radius(pts):
    """Compute the maximum radius for given circle centers."""
    # Wall clearance
    wall = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    # Pairwise min distance via KDTree (only nearest neighbor matters)
    tree = cKDTree(pts)
    d, nn = tree.query(pts, k=2)  # k=2 -> self + nearest
    pair = d[:, 1]
    # Find global min
    return min(np.min(wall), np.min(pair) / 2.0)

def _simulated_annealing_init(n_points):
    """Generate initial positions using a structured hexagonal packing."""
    # Generate a dense hexagonal lattice
    max_idx = 20
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = (i + j / 2.0) * 0.08
            y = j * (0.08 * np.sqrt(3) / 2.0)
            points.append([x, y])
    points = np.array(points)
    
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance from origin and pick central cluster
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n_points]]
    
    # Scale to fit in [-0.45, 0.45]
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 0:
        points = points * (0.45 / max_norm)
    
    return points.astype(np.float64)

def _simulated_annealing(pts, max_iter=50000, t0=0.01, t_end=0.0001, alpha=0.9997):
    """Simulated annealing to optimize circle packing."""
    n = len(pts)
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    t = t0
    for it in range(max_iter):
        # Generate random perturbation
        i = np.random.randint(n)
        dx = np.random.randn(2) * t
        new_pts = current_pts.copy()
        new_pts[i] += dx
        # Keep within bounds
        new_pts[i] = np.clip(new_pts[i], -0.49, 0.49)
        
        new_r = _compute_radius(new_pts)
        
        # Accept if better or with probability exp((new_r - current_r) / t)
        if new_r >= current_r or np.random.rand() < np.exp((new_r - current_r) / t):
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        
        # Cool down
        t = max(t * alpha, t_end)
        
        # Early stop if temperature is very low
        if t <= t_end * 1.1:
            break
    
    return best_pts, best_r

def _local_optimization(pts, max_iter=1000):
    """Local optimization using gradient-free Nelder-Mead style updates."""
    n = len(pts)
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    step_size = 0.005
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for coord in range(2):
                for direction in [-1, 1]:
                    new_pts = current_pts.copy()
                    new_pts[i, coord] += direction * step_size
                    new_pts[i, coord] = np.clip(new_pts[i, coord], -0.49, 0.49)
                    
                    new_r = _compute_radius(new_pts)
                    if new_r > current_r:
                        current_pts = new_pts
                        current_r = new_r
                        improved = True
                        if current_r > best_r:
                            best_r = current_r
                            best_pts = current_pts.copy()
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.0001:
                break
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialize with structured hexagonal packing
    x0 = _simulated_annealing_init(N)
    
    # 2. Run simulated annealing
    best_pts, best_r = _simulated_annealing(x0, max_iter=30000)
    
    # 3. Check time and run local optimization if there's time
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Run local optimization from best point
        best_pts, best_r = _local_optimization(best_pts, max_iter=500)
    
    # 4. Final validation and cleanup
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
