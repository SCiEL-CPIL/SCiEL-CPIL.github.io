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
    # If min distance is 0, score is 0
    # Simple check:
    areas = _triangle_area_vec(points)
    min_tri = np.min(areas)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback if points are collinear or few
        # If hull fails, area is likely 0 or very small
        return 0.0
        
    if hull_area <= 1e-12:
        return 0.0
        
    return min_tri / hull_area


def _make_mixture(m, k, r, phase):
    """
    Create a point set as mixture of regular m-gon (radius r) and regular k-gon (radius 1.0).
    m + k should equal N = 13.
    phase: rotation of inner polygon relative to outer.
    """
    points = []
    # Outer regular k-gon at radius 1.0
    for i in range(k):
        angle = 2 * np.pi * i / k
        points.append([np.cos(angle), np.sin(angle)])
    # Inner regular m-gon at radius r, rotated by phase
    for i in range(m):
        angle = 2 * np.pi * i / m + phase
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _optimize_partition(m, k):
    """
    Optimize over r (inner radius) and phase for partition m+k=N.
    Returns best (score, points) for this partition.
    """
    if m + k != N:
        return 0.0, None
    
    if m == 0 or k == 0:
        return 0.0, None

    best_score = 0.0
    best_pts = None
    
    # Grid search for initialization
    # r in [0, 1], phase in [0, 2*pi/m)
    # Since m and k are small, grid can be fine.
    
    # Include r=0 for the case where inner points collapse to center (valid only if m=1, else score=0)
    r_grid = np.linspace(0.0, 0.99, 20)
    # Period of phase is 2*pi / gcd(m, k)? No, the configuration is invariant under rotation of inner by 2*pi/m.
    # So phase in [0, 2*pi/m) is sufficient.
    phase_grid = np.linspace(0, 2 * np.pi / m, 20, endpoint=False)
    
    candidates = []
    
    for r in r_grid:
        for phase in phase_grid:
            pts = _make_mixture(m, k, r, phase)
            sc = _score(pts)
            candidates.append((sc, r, phase))
            
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 5 candidates with local optimization
    for sc, r0, phase0 in candidates[:5]:
        def neg_score(params):
            r, phase = params
            # Keep r in valid range
            if r < 0.0 or r > 1.0:
                return 0.0
            pts = _make_mixture(m, k, r, phase)
            return -_score(pts)
        
        # Use L-BFGS-B with bounds
        bounds = [(0.0, 1.0), (0, 2 * np.pi / m)]
        # Note: L-BFGS-B might have issues with discontinuities in the score function (min of linear functions).
        # Powell is often better for non-smooth.
        try:
            res = minimize(neg_score, [r0, phase0], method='Powell',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            r_opt, phase_opt = res.x
            r_opt = np.clip(r_opt, 0.01, 0.99)
            # Normalize phase
            phase_opt = phase_opt % (2 * np.pi / m)
            
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
    Construct 13 points.
    Compare regular 13-gon with optimized mixtures of regular polygons.
    """
    best_score = 0.0
    best_points = None
    
    # Candidate 1: Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    reg_score = _score(reg_pts)
    if reg_score > best_score:
        best_score = reg_score
        best_points = reg_pts
        
    # Candidate 2: Mixtures
    # Include m=0 (all outer) and k=0 (all inner, which is just regular 13-gon rotated/scaled, effectively same as reg_pts if r=1)
    # We allow r to go to 0 in the optimization.
    partitions = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7), (7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1), (13, 0)]
    
    for m, k in partitions:
        if m == 0:
            # All points on outer ring (k=13)
            # This is the regular 13-gon, already handled
            continue
        if k == 0:
            # All points on inner ring (m=13)
            # This is a regular 13-gon of radius r.
            # The ratio min_tri/hull_area is invariant to scaling.
            # So this is equivalent to the regular 13-gon.
            continue
            
        sc, pts = _optimize_partition(m, k)
        if pts is not None and sc > best_score:
            best_score = sc
            best_points = pts
    
    if best_points is None:
        # Fallback: regular 13-gon
        best_points = reg_pts
        
    return best_points
# EVOLVE-BLOCK-END
