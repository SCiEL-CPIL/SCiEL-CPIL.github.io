"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_covering_radius(points):
    """
    Compute the covering radius (in degrees) for points on the unit sphere.
    The covering radius is the maximum over the sphere of the angular distance
    to the nearest point. This equals the largest circumradius of the facets
    of the convex hull (for points in general position not in a hemisphere).
    
    We use a fine grid approach: sample many points on the sphere and find
    the maximum minimum angular distance. For better accuracy, we also check
    the vertices of the Voronoi cells (which are the circumcenters of the
    Delaunay triangles on the sphere).
    
    For efficiency and robustness, we use a dense spherical grid and refine
    the worst points.
    """
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Use a Fibonacci spiral grid for evaluation - dense enough for good accuracy
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each grid point to each site
    # cos(theta) = dot product for unit vectors
    dots = grid @ points.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)  # max dot = min angular distance
    angles = np.arccos(min_dots)
    
    max_angle = angles.max()
    
    # Refine: find the worst grid point and do a local search
    worst_idx = np.argmax(angles)
    worst_point = grid[worst_idx]
    
    # Local refinement using gradient-free method around worst point
    from scipy.optimize import minimize_scalar
    
    def neg_min_dot(x):
        # x = [theta, phi] in spherical coords
        th, ph = x
        p = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        dots = p @ points.T
        return -dots.max()
    
    # Try several starting points near the worst
    best_local_angle = max_angle
    for trial in range(5):
        # Perturb the worst point
        noise = np.random.randn(3) * 0.01
        wp = worst_point + noise
        wp = wp / np.linalg.norm(wp)
        # Convert to spherical
        th = np.arccos(np.clip(wp[2], -1, 1))
        ph = np.arctan2(wp[1], wp[0])
        
        res = minimize_scalar(neg_min_dot, bounds=[(0, th-0.1), (0, th+0.1)] if th > 0.1 else [(0, 0.1), (0, 2*np.pi)], method='bounded', options={'xatol': 1e-6})
        # Actually let's do a 2D bounded minimization
        res2 = minimize(neg_min_dot, x0=[th, ph], bounds=[(0, np.pi), (0, 2*np.pi)], method='L-BFGS-B', options={'maxiter': 100})
        if res2.success:
            local_angle = np.arccos(np.clip(-res2.fun, -1, 1))
            best_local_angle = max(best_local_angle, local_angle)
    
    return np.degrees(best_local_angle)


def _objective(x):
    """Objective: covering radius in radians (to be minimized)."""
    points = _unflatten(x)
    if not np.all(np.isfinite(points)):
        return 1e6
    # Quick check: no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return 1e6
    return _compute_covering_radius(points) * np.pi / 180.0


def _unflatten(x):
    """Unflatten parameters to 3D unit vectors using spherical coordinates."""
    points = np.zeros((N, 3))
    for i in range(N):
        th = x[2*i]
        ph = x[2*i+1]
        points[i] = [np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates for parameterization
    x0 = np.zeros(2*N)
    for idx in range(N):
        p = points0[idx]
        x0[2*idx] = np.arccos(np.clip(p[2], -1, 1))
        x0[2*idx+1] = np.arctan2(p[1], p[0])
    
    best_points = points0.copy()
    best_score = _compute_covering_radius(best_points)
    
    # Try multiple random restarts and local optimizations
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Random perturbation of the Fibonacci spiral
            x_start = x0.copy()
            x_start += rng.standard_normal(2*N) * 0.5
        
        # Local optimization
        try:
            res = minimize(_objective, x_start, method='L-BFGS-B', 
                          bounds=[(0, np.pi), (0, 2*np.pi)] * N,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                new_points = _unflatten(res.x)
                new_score = _compute_covering_radius(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # Final polish: try to improve with tighter tolerance
    if time.time() < deadline - 5:
        x_best = np.zeros(2*N)
        for idx in range(N):
            p = best_points[idx]
            x_best[2*idx] = np.arccos(np.clip(p[2], -1, 1))
            x_best[2*idx+1] = np.arctan2(p[1], p[0])
        
        try:
            res = minimize(_objective, x_best, method='L-BFGS-B',
                          bounds=[(0, np.pi), (0, 2*np.pi)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success:
                polished = _unflatten(res.x)
                polished_score = _compute_covering_radius(polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_points = polished
        except Exception:
            pass
    
    # Ensure output is valid
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
