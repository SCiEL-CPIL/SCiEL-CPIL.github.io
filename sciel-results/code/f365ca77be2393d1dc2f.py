"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13
TIME_LIMIT = 110  # seconds, safety margin before 120s kill

def _compute_score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    
    # Convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_compute_score(points)

def _is_convex(points):
    """Check if points form a convex polygon (all points on hull)."""
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False

def _local_optimize(points, time_budget):
    """Local optimization using Nelder-Mead or similar."""
    points = np.asarray(points, dtype=np.float64)
    x0 = points.flatten()
    
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalize non-convex configurations
        if not _is_convex(pts):
            return 1e6 - _compute_score(pts)
        return -_compute_score(pts)
    
    # Use Nelder-Mead
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    elapsed = time.time() - start_time
    best_pts = result.x.reshape(-1, 2)
    best_score = -result.fun if result.fun < 1e5 else _compute_score(best_pts)
    
    return best_pts, best_score, elapsed

def _random_convex_points(rng, n):
    """Generate random points in convex position."""
    # Generate random angles sorted, then random radii
    angles = np.sort(rng.uniform(0, 2*np.pi, n))
    # Ensure no two angles are too close
    radii = rng.uniform(0.5, 1.0, n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _perturb(points, sigma, rng):
    """Add Gaussian noise."""
    noise = rng.normal(0, sigma, size=points.shape)
    return points + noise

def construct():
    start_time = time.time()
    rng = np.random.default_rng(N)
    
    best_points = None
    best_score = -float('inf')
    
    # Try several initial configurations
    n_initial = 8
    for init_idx in range(n_initial):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        if init_idx == 0:
            # Regular polygon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif init_idx == 1:
            # Slightly perturbed regular polygon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = 1.0 + rng.normal(0, 0.05, N)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif init_idx == 2:
            # Random convex points
            points = _random_convex_points(rng, N)
        elif init_idx == 3:
            # Another random convex
            points = _random_convex_points(rng, N)
        elif init_idx == 4:
            # Regular polygon with alternating radii
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.where(np.arange(N) % 2 == 0, 1.0, 0.9)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif init_idx == 5:
            # Another perturbation
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = 1.0 + rng.normal(0, 0.1, N)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif init_idx == 6:
            # Random
            points = _random_convex_points(rng, N)
        else:
            # Random
            points = _random_convex_points(rng, N)
        
        # Ensure convexity
        if not _is_convex(points):
            # Project to convex hull vertices
            try:
                hull = ConvexHull(points)
                points = points[hull.vertices]
                # If we lost points, just use the hull
                if len(points) < N:
                    # Re-generate
                    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                    points = np.column_stack([np.cos(angles), np.sin(angles)])
            except Exception:
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Local optimization
        time_budget = TIME_LIMIT - (time.time() - start_time) - 15
        if time_budget < 5:
            break
        
        try:
            opt_points, opt_score, elapsed = _local_optimize(points, time_budget)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # If no valid points found, fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
