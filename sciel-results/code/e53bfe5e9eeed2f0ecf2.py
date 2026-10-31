"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute all triples of indices for the 11 points
# We need to consider all C(11,3) = 165 triangles
import itertools
TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLES_ARR = np.array(TRIPLES)  # shape (165, 3)


def _area_from_points(pts):
    """Compute areas of all 165 triangles given 11 points. pts shape (11,2)."""
    # pts[i] for i in 0..10
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    p0 = pts[TRIPLES_ARR[:, 0, 0]]  # (165, 2)
    p1 = pts[TRIPLES_ARR[:, 1, 0]]
    p2 = pts[TRIPLES_ARR[:, 2, 0]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    return 0.5 * np.abs(cross)


def _neg_min_area(pts):
    """Negative of minimum triangle area (for minimization)."""
    areas = _area_from_points(pts)
    return -np.min(areas)


def _in_triangle_check(pts):
    """Check if all points are inside or on the boundary of the reference triangle."""
    # For point p, check barycentric coordinates relative to V0, V1, V2
    # A point is inside if all barycentric coords are in [0, 1]
    # Using edge-based method: for each edge, check point is on the correct side
    eps = 1e-9
    for p in pts:
        # Edge V0->V1: normal pointing inward is (0, 1) roughly, but let's use cross product
        # For edge from A to B, point P is on the left side if cross(B-A, P-A) >= 0 (for CCW triangle)
        # Triangle V0, V1, V2 is CCW
        # Edge V0->V1: cross(V1-V0, P-V0) should be >= 0
        c0 = np.cross(V1 - V0, p - V0)
        c1 = np.cross(V2 - V1, p - V1)
        c2 = np.cross(V0 - V2, p - V2)
        if c0 < -eps or c1 < -eps or c2 < -eps:
            return False
    return True


def _project_to_triangle(p):
    """Project point p onto the convex hull (triangle) if it's outside."""
    # Simple approach: if outside, move towards centroid
    centroid = (V0 + V1 + V2) / 3
    # This is a rough projection; for optimization we use bounds instead
    return p


def _penalty(pts):
    """Penalty for points outside the triangle."""
    penalty = 0.0
    for p in pts:
        c0 = np.cross(V1 - V0, p - V0)
        c1 = np.cross(V2 - V1, p - V1)
        c2 = np.cross(V0 - V2, p - V2)
        if c0 < 0:
            penalty += c0 ** 2
        if c1 < 0:
            penalty += c1 ** 2
        if c2 < 0:
            penalty += c2 ** 2
    return penalty


def _objective(pts_flat):
    """Objective function for optimization: minimize negative min area + penalty."""
    pts = pts_flat.reshape(N, 2)
    min_area = np.min(_area_from_points(pts))
    pen = _penalty(pts)
    return -min_area + 100.0 * pen


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_pts = None
    best_score = -np.inf
    
    # Initial random starts
    np.random.seed(42)
    
    # Try a few initial configurations
    initial_configs = []
    
    # Config 1: Random
    for seed in range(20):
        np.random.seed(seed)
        # Generate random points in triangle using rejection sampling
        pts = []
        while len(pts) < N:
            u, v = np.random.random(2)
            if u + v < 1:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                pts.append(p)
        initial_configs.append(np.array(pts))
    
    # Config 2: Grid-like
    # Place points in a structured way
    grid_pts = []
    # 3x3 grid in barycentric coordinates
    for i in range(3):
        for j in range(3):
            if i + j < 3:
                a = i / 2.0
                b = j / 2.0
                c = 1 - a - b
                p = a * V0 + b * V1 + c * V2
                grid_pts.append(p)
    # We need 11 points, we have 6 from 3x3 with i+j<3
    # Add more
    np.random.seed(0)
    while len(grid_pts) < N:
        u, v = np.random.random(2)
        if u + v < 1:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            grid_pts.append(p)
    initial_configs.append(np.array(grid_pts[:N]))
    
    # Config 3: Vertices and edge midpoints
    structured = [V0, V1, V2,
                  (V0 + V1) / 2, (V1 + V2) / 2, (V2 + V0) / 2,
                  (V0 + V1 + V2) / 3]
    # Add 4 more
    np.random.seed(1)
    while len(structured) < N:
        u, v = np.random.random(2)
        if u + v < 1:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            structured.append(p)
    initial_configs.append(np.array(structured[:N]))
    
    # Evaluate initial configs
    for pts in initial_configs:
        if _in_triangle_check(pts):
            score = np.min(_area_from_points(pts))
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = initial_configs[0].copy()
        best_score = np.min(_area_from_points(best_pts))
    
    # Local optimization from best initial config
    # Use Nelder-Mead or BFGS
    x0 = best_pts.flatten()
    
    # Run multiple optimization attempts from different starting points
    attempts = 0
    for init_idx, pts_init in enumerate(initial_configs[:10]):
        if time.time() > deadline - 5.0:
            break
        
        x0 = pts_init.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:  # result.fun is -min_area + penalty
                pts_opt = result.x.reshape(N, 2)
                if _in_triangle_check(pts_opt):
                    score = np.min(_area_from_points(pts_opt))
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
        except Exception:
            pass
        
        attempts += 1
        
        # If we have time, try more optimizations
        if time.time() > deadline - 10.0:
            break
    
    # If we still have time, do a few more random restarts with optimization
    while time.time() < deadline - 5.0:
        np.random.seed(int(time.time() * 1000) % 10000)
        pts = []
        while len(pts) < N:
            u, v = np.random.random(2)
            if u + v < 1:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                pts.append(p)
        pts = np.array(pts)
        
        if _in_triangle_check(pts):
            x0 = pts.flatten()
            try:
                result = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.success:
                    pts_opt = result.x.reshape(N, 2)
                    if _in_triangle_check(pts_opt):
                        score = np.min(_area_from_points(pts_opt))
                        if score > best_score:
                            best_score = score
                            best_pts = pts_opt.copy()
            except Exception:
                pass
    
    return best_pts

