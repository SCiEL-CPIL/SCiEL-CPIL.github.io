# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(13,3)=286 triangles."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn score: A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return -1.0
    return a_min / a_hull


def _objective(params):
    """Objective to minimize (negative score) for SLSQP."""
    points = params.reshape(13, 2)
    # Ensure points are within reasonable bounds
    if not np.all(np.isfinite(points)):
        return 1e10
    score = _score(points)
    if score < 0:
        return 1e10
    return -score


def _make_seed_hexagonal():
    """Create a hexagonal/triangular lattice seed for 13 points."""
    # Triangular lattice: center + rings
    # For 13 points, use a hexagonal arrangement
    points = np.zeros((13, 2))
    # Center
    points[0] = [0.0, 0.0]
    
    # Ring 1: 6 points at distance 1
    for i in range(6):
        angle = i * np.pi / 3
        points[i + 1] = [np.cos(angle), np.sin(angle)]
    
    # Ring 2: 6 points at distance 2 (but only 6 more to get 13 total)
    # Actually, let's use a different approach: triangular grid subset
    # Use points from a triangular lattice
    pts = []
    # Row 0: 1 point
    pts.append([0.0, 0.0])
    # Row 1: 2 points
    pts.append([-0.5, np.sqrt(3)/2])
    pts.append([0.5, np.sqrt(3)/2])
    # Row 2: 3 points
    pts.append([-1.0, np.sqrt(3)])
    pts.append([0.0, np.sqrt(3)])
    pts.append([1.0, np.sqrt(3)])
    # Row 3: 4 points
    pts.append([-1.5, 1.5*np.sqrt(3)])
    pts.append([-0.5, 1.5*np.sqrt(3)])
    pts.append([0.5, 1.5*np.sqrt(3)])
    pts.append([1.5, 1.5*np.sqrt(3)])
    # Row 4: 3 points (to make 13 total: 1+2+3+4+3=13)
    pts.append([-1.0, 2*np.sqrt(3)])
    pts.append([0.0, 2*np.sqrt(3)])
    pts.append([1.0, 2*np.sqrt(3)])
    
    points = np.array(pts)
    # Center and normalize
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_grid():
    """Create a grid-based seed for 13 points."""
    # 4x4 grid minus one corner = 15, use 13
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    # Take first 13
    points = np.array(pts[:13], dtype=float)
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_circular():
    """Create a circular seed with center point."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[i + 1] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_mixed():
    """Create a mixed seed: center + inner ring + outer ring."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    # Inner ring: 4 points at distance 0.5
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        points[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    # Outer ring: 8 points at distance 1.0
    for i in range(8):
        angle = i * 2 * np.pi / 8
        points[i + 5] = [np.cos(angle), np.sin(angle)]
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses structured seeds with SLSQP optimization.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Generate candidate seeds
    seeds = [
        _make_seed_hexagonal(),
        _make_seed_grid(),
        _make_seed_circular(),
        _make_seed_mixed(),
    ]
    
    # Also add some randomized perturbations of the best seeds
    rng = np.random.default_rng(seed=123)
    for seed_pts in seeds[:2]:
        for _ in range(3):
            perturbed = seed_pts + rng.normal(0, 0.05, (13, 2))
            seeds.append(perturbed)
    
    best_points = None
    best_score = -1.0
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        # Normalize seed to unit scale
        seed = seed - seed.mean(axis=0)
        scale = np.max(np.abs(seed))
        if scale > 0:
            seed = seed / scale
        
        initial_params = seed.flatten()
        
        # SLSQP optimization
        try:
            result = minimize(
                _objective,
                initial_params,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8},
            )
            
            if result.success or result.fun < -best_score:
                opt_points = result.x.reshape(13, 2)
                # Normalize: center and scale
                opt_points = opt_points - opt_points.mean(axis=0)
                sc = np.max(np.abs(opt_points))
                if sc > 0:
                    opt_points = opt_points / sc
                
                score = _score(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
                    
        except Exception:
            continue
    
    # If optimization didn't find anything good, use the best seed
    if best_points is None:
        best_points = seeds[0]
        best_score = _score(best_points)
    
    # Final verification
    if not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback to a simple valid configuration
        best_points = _make_seed_hexagonal()
    
    return best_points


# EVOLVE-BLOCK-END
