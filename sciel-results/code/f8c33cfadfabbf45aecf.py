"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _triangle_area(p1, p2, p3):
    """Area of triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    """Compute convex hull area of a set of 2D points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / convex hull area. Higher is better."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf'):
        return 0.0
    
    return min_area / hull_area

def _neg_objective(x):
    """Negative score for minimization."""
    points = x.reshape(-1, 2)
    return -_score(points)

def _regularize(points, eps=1e-6):
    """Add tiny perturbation to avoid degeneracy."""
    return points + eps * np.random.randn(*points.shape)

def construct():
    best_points = None
    best_score = -1.0
    
    # Try multiple initial configurations
    n_starts = 20
    
    for start_idx in range(n_starts):
        if _time_left() < 5.0:
            break
        
        if start_idx == 0:
            # Regular grid-like initial guess
            points = _regularize(_initial_grid(N))
        elif start_idx == 1:
            # Circle
            points = _regularize(_initial_circle(N))
        elif start_idx == 2:
            # Random
            rng = np.random.RandomState(start_idx * 12345)
            points = rng.rand(N, 2)
        else:
            # Random with some structure
            rng = np.random.RandomState(start_idx * 777 + 42)
            points = rng.rand(N, 2)
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                _neg_objective,
                points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # Final polish: run more iterations on the best point
    if best_points is not None and _time_left() > 10.0:
        try:
            result = minimize(
                _neg_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return random points
        rng = np.random.RandomState(42)
        best_points = rng.rand(N, 2)
    
    return best_points

def _initial_grid(n):
    """Create a grid-like initial configuration."""
    # Arrange points in a roughly square grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([(j + 0.5) / cols, (i + 0.5) / rows])
    return np.array(points[:n])

def _initial_circle(n):
    """Create points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points
# EVOLVE-BLOCK-END
