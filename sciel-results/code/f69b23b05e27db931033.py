"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _score_from_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute all triangle areas
    # Number of triples: n*(n-1)*(n-2)/6
    # For n=14: 364 triangles
    min_area = float('inf')
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx1 = xj - xi
            dy1 = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs(dx1 * (yk - yi) - dy1 * (xk - xi))
                if area < min_area:
                    min_area = area
    if min_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _params_to_points(params):
    """Convert polar parameters (radius, angle) pairs to Cartesian points.
    
    params: shape (2*N,) alternating r, theta for each point
    We sort by angle to ensure a consistent ordering, then place on disk.
    """
    n = N
    radii = params[0::2]
    angles = params[1::2]
    
    # Sort by angle to get a circular ordering
    order = np.argsort(angles)
    radii_sorted = radii[order]
    angles_sorted = angles[order]
    
    # Convert to Cartesian
    pts = np.column_stack([
        radii_sorted * np.cos(angles_sorted),
        radii_sorted * np.sin(angles_sorted)
    ])
    return pts


def _objective(params):
    """Objective function for differential evolution (to minimize)."""
    pts = _params_to_points(params)
    score = _score_from_points(pts)
    return -score  # Negate because we maximize


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Bounds for differential evolution
    # Radii in [0.1, 1.0], angles in [0, 2*pi)
    bounds = [(0.1, 1.0), (0, 2 * np.pi)] * N
    
    best_score = 0.0
    best_params = None
    
    # Try multiple restarts with different random seeds
    rng = np.random.RandomState(N)
    
    for trial in range(5):
        if time.time() > deadline - 30:  # Need at least 30s for one run
            break
        
        # Set seed for reproducibility within trial
        np.random.seed(N + trial * 1000)
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=N + trial * 1000,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                polish=False,
                init='sobol',
                mutation=(0.3, 0.9),
                recombination=0.7
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is not None:
        pts = _params_to_points(best_params)
        # Normalize by convex hull area to get final score
        final_score = _score_from_points(pts)
        return pts
    
    # Fallback: uniform random points
    rng = np.random.RandomState(N)
    pts = rng.random((N, 2))
    return pts
# EVOLVE-BLOCK-END
