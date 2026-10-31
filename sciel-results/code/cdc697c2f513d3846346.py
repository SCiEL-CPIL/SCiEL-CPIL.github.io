"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas_vectorized(points):
    """Compute areas of all triangles formed by triples of points using vectorized ops."""
    n = len(points)
    # points shape (n, 2)
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # We compute all pairs of vectors and then cross products
    # To be efficient, we can use broadcasting for a subset or compute all
    # For n=13, total triples = 286, which is small. Let's just loop over triples but vectorize the area calc.
    
    # Actually, for n=13, a simple loop is fast enough. But let's try to be a bit smarter.
    # areas[i,j,k] for i<j<k
    areas = np.full((n, n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                # Cross product in 2D: v1x*v2y - v1y*v2x
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[i, j, k] = 0.5 * abs(cross)
    return areas

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    # We only need the minimum, so we can stop early if we find 0, but usually we compute all.
    min_area = np.inf
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
    return min_area / hull_area

def _normalize(points):
    """
    Normalize points so that their convex hull has unit area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        center = np.mean(points, axis=0)
        return points - center
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    return (points - center) * scale

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])

def _build_symmetric_config(n):
    """
    Create a configuration with some symmetry.
    For 13 points: 1 center, 6 on inner ring, 6 on outer ring? Or just regular polygon.
    Let's try: 1 center, 12 on two rings of 6?
    Or 1 center, 12 on one ring?
    """
    # 1 center, 12 on a circle
    center = np.array([[0.0, 0.0]])
    ring = _regular_polygon(12, radius=1.0)
    return np.vstack([center, ring])

def _build_template_13_0():
    """Regular 13-gon"""
    return _regular_polygon(13)

def _build_template_7_6():
    """7 on outer ring, 6 on inner ring (concentric)"""
    outer = _regular_polygon(7, radius=1.0)
    inner = _regular_polygon(6, radius=0.5)
    # Offset inner ring to avoid alignment?
    # Rotate inner by pi/6
    angle = np.pi / 6
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    return np.vstack([outer, inner])

def _build_template_6_7():
    """6 on outer ring, 7 on inner ring"""
    outer = _regular_polygon(6, radius=1.0)
    inner = _regular_polygon(7, radius=0.6)
    # Rotate inner
    angle = np.pi / 7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    return np.vstack([outer, inner])

def _build_template_5_4_4():
    """5 center-ish, 4 inner, 4 outer? No, 5+4+4=13.
    1 center, 4 inner, 4 outer, 4 more?
    Let's do 1 center, 6 inner, 6 outer.
    """
    center = np.array([[0.0, 0.0]])
    inner = _regular_polygon(6, radius=0.5)
    outer = _regular_polygon(6, radius=1.0)
    # Rotate outer by pi/6 to interleave
    angle = np.pi / 6
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    outer = outer @ rot.T
    return np.vstack([center, inner, outer])

def _build_template_8_5():
    """8 outer, 5 inner"""
    outer = _regular_polygon(8, radius=1.0)
    inner = _regular_polygon(5, radius=0.6)
    # Rotate inner
    angle = np.pi / 5
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inner = inner @ rot.T
    return np.vstack([outer, inner])

def _soft_min_objective(x, deadline):
    """
    Objective: minimize -log(sum(exp(-areas/T))) approximated by min area.
    Actually, just minimize -score.
    """
    if time.time() > deadline:
        return 1e10
    
    points = x.reshape(N, 2)
    points_norm = _normalize(points)
    score = _score(points_norm)
    if score <= 0:
        return 1e10
    return -score

def _hard_objective(x, deadline):
    """
    Objective: minimize -min_area + penalty * (hull_area - 1)^2 ?
    No, score is ratio.
    Let's stick to _soft_min_objective which is -score.
    """
    return _soft_min_objective(x, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    def update_best(points):
        nonlocal best_points, best_score
        points = _normalize(points)
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score
    
    # Initial candidates
    templates = [
        _build_template_13_0,
        _build_template_7_6,
        _build_template_6_7,
        _build_template_5_4_4,
        _build_template_8_5,
        _build_symmetric_config
    ]
    
    for builder in templates:
        if time.time() > deadline:
            break
        try:
            pts = builder()
            update_best(pts)
        except Exception:
            pass
    
    # Add some random starts
    for _ in range(5):
        if time.time() > deadline:
            break
        pts = rng.uniform(-1, 1, (N, 2))
        update_best(pts)
    
    # Local search using Nelder-Mead
    # We need to optimize the 26 coordinates.
    # We will run Nelder-Mead from several best starting points.
    
    # Get top starting points
    starts = []
    if best_points is not None:
        starts.append(best_points)
    
    # Also try perturbing the best point
    if best_points is not None:
        for _ in range(3):
            perturbed = best_points + rng.normal(0, 0.1, (N, 2))
            starts.append(perturbed)
    
    # Run optimization
    for init_pts in starts:
        if time.time() > deadline:
            break
        
        init_pts = _normalize(init_pts)
        init_flat = init_pts.flatten()
        
        try:
            result = minimize(
                _hard_objective,
                init_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        if result.fun < -best_score:
            new_points = _normalize(result.x.reshape(N, 2))
            update_best(new_points)
    
    # If we have time, do a refined search
    if best_points is not None and time.time() < deadline - 10:
        init_flat = best_points.flatten()
        try:
            result = minimize(
                _hard_objective,
                init_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
        except Exception:
            pass
        
        if result.fun < -best_score:
            new_points = _normalize(result.x.reshape(N, 2))
            update_best(new_points)
    
    if best_points is None:
        return _regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
