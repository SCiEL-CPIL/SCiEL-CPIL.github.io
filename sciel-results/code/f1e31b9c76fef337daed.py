"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(pts):
    """Compute the minimum radius for a set of centers."""
    n = len(pts)
    # Minimum distance to boundary
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    # Minimum pairwise distance
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d < min_pair_dist:
                min_pair_dist = d
    return min(min_pair_dist / 2.0, np.min(dists_boundary))


def _hex_lattice_init(n_points):
    """Generate initial hexagonal lattice configuration."""
    # Start with a reasonable spacing
    spacing = 0.12
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n_points:
            # Select the n_points closest to center
            idx = np.argsort(radius, kind="stable")[:n_points]
            return points[idx]
        spacing *= 0.95


def _slsqp_optimize(pts, deadline):
    """Optimize using SLSQP to maximize minimum radius."""
    n = len(pts)
    x0 = pts.flatten()
    
    def objective(x):
        pts_x = x.reshape(n, 2)
        # Return negative of minimum radius (to maximize)
        r = _compute_score(pts_x)
        return -r
    
    # Constraints: all points must be inside unit circle
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - np.hypot(x[2*i], x[2*i+1])})
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    start_time = time.time()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})
        if result.success and _compute_score(result.x.reshape(n, 2)) > _compute_score(pts):
            return result.x.reshape(n, 2)
    except Exception:
        pass
    
    return pts


def _local_polish(pts, deadline):
    """Local polishing using gradient-free optimization on the minimum radius."""
    n = len(pts)
    current_pts = pts.copy()
    current_score = _compute_score(current_pts)
    
    # Simple local search: try small perturbations
    step_size = 0.01
    for _ in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    test_pts = current_pts.copy()
                    test_pts[i, 0] += dx
                    test_pts[i, 1] += dy
                    
                    # Check if still valid (inside circle)
                    if np.hypot(test_pts[i, 0], test_pts[i, 1]) < 1.0:
                        test_score = _compute_score(test_pts)
                        if test_score > current_score + 1e-12:
                            current_pts = test_pts
                            current_score = test_score
                            improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return current_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Step 1: Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    initial_score = _compute_score(pts)
    
    # Step 2: SLSQP optimization
    if time.time() < deadline - 30.0:
        pts = _slsqp_optimize(pts, deadline)
    
    # Step 3: Local polish
    if time.time() < deadline - 5.0:
        pts = _local_polish(pts, deadline)
    
    # Ensure validity
    final_score = _compute_score(pts)
    if final_score <= 0:
        # Fallback to initial lattice
        pts = _hex_lattice_init(N)
    
    return pts
# EVOLVE-BLOCK-END
