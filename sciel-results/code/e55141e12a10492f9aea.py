"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    # pts: (N, 2)
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _points_from_radial(radial):
    """Convert radial offsets to 2D points on a unit circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    pts = np.column_stack([radial * cos_a, radial * sin_a])
    return pts

def _soft_min_radial_objective(radial, beta=100.0):
    """Soft-min of triangle areas / hull area, negative for minimization.
    radial: (N,) array of radial distances."""
    pts = _points_from_radial(radial)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    # Soft min using log-sum-exp trick
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _soft_min_full_objective(x, beta=100.0):
    """Soft-min of triangle areas / hull area, negative for minimization.
    x: (2N,) flattened point coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _halton_points(seed=0):
    """Generate points using Halton sequence mapped to unit disk."""
    rng = np.random.RandomState(seed)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    # Use multiple bases for 2D
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i, 0] = 2 * halton(i + 1, 2) - 1
        pts[i, 1] = 2 * halton(i + 1, 3) - 1
    # Scale to unit disk
    pts *= 0.9
    return pts

def _regular_polygon_pts(seed=0):
    """Generate regular polygon points with slight perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radial = np.ones(N) + rng.normal(0, 0.05, N)
    pts = _points_from_radial(radial)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = 0.0
    best_pts = _regular_polygon_pts(0)
    
    # Phase 1: Optimize radial parameters (13D)
    for seed in range(30):
        if time.time() > deadline - 20:
            break
        
        if seed % 2 == 0:
            # Start from regular polygon with perturbation
            rng = np.random.RandomState(seed)
            radial0 = np.ones(N) + rng.normal(0, 0.1, N)
        else:
            # Start from Halton-based radial estimates
            pts_h = _halton_points(seed)
            # Estimate radial distances
            radial0 = np.sqrt(pts_h[:, 0]**2 + pts_h[:, 1]**2)
            radial0 = np.clip(radial0, 0.5, 1.5)
        
        try:
            result = minimize(
                _soft_min_radial_objective,
                radial0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            radial_opt = result.x
            pts = _points_from_radial(radial_opt)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Full 2D polish from best radial solution
    if time.time() < deadline - 15:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _soft_min_full_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Phase 3: Additional multi-start full 2D optimization
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        
        x0 = _halton_points(seed).flatten()
        try:
            result = minimize(
                _soft_min_full_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-8}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END
