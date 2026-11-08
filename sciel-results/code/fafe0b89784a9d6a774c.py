import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)
_CENTROID = np.array([0.5, _H / 3.0])

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _is_inside_triangle(x, y):
    """Check if point (x, y) is inside the equilateral triangle."""
    return (y >= 0) and (y <= _SQRT3 * x) and (y <= _SQRT3 - _SQRT3 * x)


def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses barycentric coordinates for robust projection.
    """
    pts = np.array(pts, dtype=float)
    
    # Vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, _H])
    
    # Precompute for barycentric
    v0 = B - A
    v1 = C - A
    v2 = A - A  # zero
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d02 = np.zeros(len(pts))
    d12 = np.zeros(len(pts))
    
    denom = d00 * d11 - d01 * d01
    if denom < 1e-12:
        return pts
    
    # For each point, compute barycentric coordinates
    for i in range(len(pts)):
        v = pts[i] - A
        d0 = np.dot(v0, v)
        d1 = np.dot(v1, v)
        
        # Barycentric coordinates
        # a = (d11*d0 - d01*d1) / denom
        # b = (d00*d1 - d01*d0) / denom
        # c = 1 - a - b
        a = (d11 * d0 - d01 * d1) / denom
        b = (d00 * d1 - d01 * d0) / denom
        c = 1.0 - a - b
        
        # Clamp to [0, 1]
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        c = np.clip(c, 0.0, 1.0)
        
        # Normalize if needed
        s = a + b + c
        if s < 1e-12:
            a = b = c = 1.0 / 3.0
        else:
            a /= s
            b /= s
            c /= s
        
        pts[i] = a * A + b * B + c * C
    
    return pts


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _objective(flat):
    """
    Objective function for differential evolution.
    Returns negative minimum triangle area (to minimize).
    """
    pts = _project_to_triangle(flat.reshape(N, 2))
    return -_min_triangle_area(pts)


def _make_symmetric_initial():
    """
    Create an initial configuration with 3-fold rotational symmetry.
    11 points: 1 at center, 3 on each of 3 orbits (3*3+1 = 10... need 11).
    Let's do: 1 center, 2 on each of 3 orbits? No, 1 + 3*2 = 7.
    1 center + 3 + 3 + 3 + 1? 
    Let's just use a structured grid-like initial point set.
    """
    pts = []
    # Center
    pts.append(_CENTROID.copy())
    
    # 3-fold symmetric orbits
    # Orbit 1: 3 points at radius r1, angles 0, 120, 240 (rotated by centroid)
    r1 = 0.15
    for k in range(3):
        angle = k * 2 * np.pi / 3
        # In triangle coordinates, we need to be careful
        # Let's just place them and project
        p = _CENTROID + r1 * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    
    # Orbit 2: 3 points at radius r2
    r2 = 0.3
    for k in range(3):
        angle = k * 2 * np.pi / 3 + np.pi / 3
        p = _CENTROID + r2 * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    
    # Orbit 3: 3 points at radius r3
    r3 = 0.1
    for k in range(3):
        angle = k * 2 * np.pi / 3 + np.pi / 6
        p = _CENTROID + r3 * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    
    # One more point to make 11
    pts.append(_CENTROID + np.array([0.0, 0.05]))
    
    pts = np.array(pts)
    return _project_to_triangle(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Try differential evolution
    try:
        # Bounds: x in [0, 1], y in [0, H]
        bounds = [(0.0, 1.0)] * N + [(0.0, _H)] * N
        
        # We'll do a quick DE run
        # Use a moderate population size and max iterations
        # DE can be slow, so we limit it
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result.success or result.fun < -best_score:
            pts = _project_to_triangle(result.x.reshape(N, 2))
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    except Exception:
        pass
    
    # If DE didn't find anything good, try a simpler approach
    if best_points is None or best_score < 0.01:
        # Use the symmetric initial and refine with Nelder-Mead
        try:
            from scipy.optimize import minimize
            
            init_pts = _make_symmetric_initial().flatten()
            
            def objective_nm(flat):
                pts = _project_to_triangle(flat.reshape(N, 2))
                return -_min_triangle_area(pts)
            
            res = minimize(objective_nm, init_pts, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_pts = _project_to_triangle(res.x.reshape(N, 2))
            score = _min_triangle_area(final_pts)
            
            if score > best_score:
                best_score = score
                best_points = final_pts
        except Exception:
            pass
    
    # Fallback: random search
    if best_points is None:
        rng = np.random.default_rng(42)
        best_score = -np.inf
        for _ in range(50):
            if time.time() > deadline:
                break
            pts = rng.random((N, 2)) * [1.0, _H]
            pts = _project_to_triangle(pts)
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    if best_points is None:
        # Last resort: uniform grid
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, _H],
            [0.5, 0.0], [0.25, _H / 2], [0.75, _H / 2],
            [0.5, _H / 3], [0.5, 2 * _H / 3],
            [0.33, _H / 3], [0.67, _H / 3], [0.5, _H / 2]
        ])
        best_points = _project_to_triangle(pts)
    
    return best_points
