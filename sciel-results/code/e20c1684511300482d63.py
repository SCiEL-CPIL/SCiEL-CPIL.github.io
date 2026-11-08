"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _triangle_area_vec(points):
    """
    Compute areas of all triangles formed by triples of points.
    points: (N, 2) array.
    Returns: array of areas.
    """
    n = len(points)
    # Get all combinations of 3 indices
    idx = np.array(list(combinations(range(n), 3)))
    # idx shape: (M, 3)
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Cross product in 2D: (p2-p1) x (p3-p1)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas


def _score(points):
    """Compute score: min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    # Check for duplicate points (degenerate triangles)
    areas = _triangle_area_vec(points)
    min_tri = np.min(areas)
    
    if min_tri < 1e-9:
        return 0.0
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback if points are collinear or few
        return 0.0
        
    if hull_area <= 1e-12:
        return 0.0
        
    return min_tri / hull_area


def _make_mixture(m, k, r, phase):
    """
    Create a point set as mixture of regular m-gon (radius r) and regular k-gon (radius 1.0).
    m + k should equal N = 13.
    phase: rotation of the m-gon relative to the k-gon.
    """
    points = []
    # Regular k-gon at radius 1.0
    if k > 0:
        for i in range(k):
            angle = 2 * np.pi * i / k
            points.append([np.cos(angle), np.sin(angle)])
    # Regular m-gon at radius r, rotated by phase
    if m > 0:
        for i in range(m):
            angle = 2 * np.pi * i / m + phase
            points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _optimize_partition(m, k):
    """
    Optimize over r (radius of m-gon) and phase for partition m+k=N.
    The k-gon is fixed at radius 1.0.
    Returns best (score, points) for this partition.
    """
    if m + k != N:
        return 0.0, None
    
    # Handle edge cases where one polygon is empty
    if m == 0:
        # Just k-gon
        pts = _make_mixture(0, k, 0, 0)
        return _score(pts), pts
    if k == 0:
        # Just m-gon (radius r can be anything, shape is regular m-gon)
        # We can just return a regular m-gon at radius 1
        pts = _make_mixture(m, 0, 1.0, 0)
        return _score(pts), pts

    best_score = 0.0
    best_pts = None
    
    # We need to search r and phase.
    # r can be any positive value. The hull will be the convex hull of the combined points.
    # If r < 1, hull is likely k-gon (unless m-gon sticks out? No, m-gon is inside).
    # If r > 1, hull is likely m-gon.
    # So we should search r in a range that covers both, e.g., [0.1, 2.0].
    
    # Phase period is 2*pi/m.
    phase_period = 2 * np.pi / m
    
    # Multi-start optimization
    # 1. Coarse grid
    r_grid = np.linspace(0.1, 2.0, 30)
    phase_grid = np.linspace(0, phase_period, 20, endpoint=False)
    
    candidates = []
    for r in r_grid:
        for phase in phase_grid:
            pts = _make_mixture(m, k, r, phase)
            sc = _score(pts)
            candidates.append((sc, r, phase))
            
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 10 candidates
    for sc, r0, phase0 in candidates[:10]:
        def neg_score(params):
            r, phase = params
            # Bounds for r to avoid degeneracy or huge values
            if r < 0.05 or r > 5.0:
                return 0.0
            pts = _make_mixture(m, k, r, phase)
            return -_score(pts)
        
        bounds = [(0.05, 5.0), (0, phase_period)]
        
        try:
            # Powell is good for non-smooth
            res = minimize(neg_score, [r0, phase0], method='Powell',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            r_opt, phase_opt = res.x
            r_opt = np.clip(r_opt, 0.05, 5.0)
            phase_opt = phase_opt % phase_period
            
            pts = _make_mixture(m, k, r_opt, phase_opt)
            sc_opt = _score(pts)
            
            if sc_opt > best_score:
                best_score = sc_opt
                best_pts = pts
        except Exception:
            pass
            
    return best_score, best_pts


def construct():
    """
    Construct 13 points as a weighted mixture of two regular polygons.
    Try all partitions m+k=13 and optimize radii/phases.
    """
    best_score = 0.0
    best_points = None
    
    # Include all partitions including (0, 13) and (13, 0)
    partitions = [(m, N - m) for m in range(N + 1)]
    
    for m, k in partitions:
        sc, pts = _optimize_partition(m, k)
        if pts is not None and sc > best_score:
            best_score = sc
            best_points = pts
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
