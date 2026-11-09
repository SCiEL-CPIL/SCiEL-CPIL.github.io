import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)
_N_TRIPLES = len(_i_list)

def _compute_areas(points):
    """Vectorized triangle areas for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    areas = _compute_areas(points)
    ma = np.min(areas)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def _score_neg(points):
    return -get_score(points)

def _regular_polygon(radius=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def _affine_to_points(params):
    """
    params: [a, b, c, d, e, f]
    Point = [[a, b], [c, d]] @ [x, y] + [e, f]
    where [x, y] are vertices of regular 14-gon
    """
    a, b, c, d, e, f = params
    base = _regular_polygon(1.0)
    # Apply affine transform
    x_new = a * base[:, 0] + b * base[:, 1] + e
    y_new = c * base[:, 0] + d * base[:, 1] + f
    return np.column_stack((x_new, y_new))

def _radial_ang_to_points(params):
    """
    params: [r_0..r_13, theta_1..theta_13]
    theta_0 is fixed to 0.
    """
    radii = params[:N]
    angles = np.zeros(N)
    angles[0] = 0.0
    if N > 1:
        angles[1:] = params[N:]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Regular Polygon
    best_points = _regular_polygon(1.0)
    best_score = get_score(best_points)
    
    # Strategy 1: Affine Transformation Global Search
    # Bounds for affine params
    # Regular polygon has radius 1. 
    # We want to allow significant distortion.
    bounds_affine = [
        (-2, 2),  # a
        (-2, 2),  # b
        (-2, 2),  # c
        (-2, 2),  # d
        (-2, 2),  # e
        (-2, 2)   # f
    ]
    
    def obj_affine(params):
        if time.time() > deadline - 5.0:
            return 0.0
        pts = _affine_to_points(params)
        return _score_neg(pts)
        
    try:
        result_affine = differential_evolution(
            obj_affine, 
            bounds_affine, 
            maxiter=50, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            init='sobol'
        )
        if result_affine.fun < 0:
            pts = _affine_to_points(result_affine.x)
            score = -result_affine.fun
            if score > best_score:
                best_score = score
                best_points = pts
    except Exception:
        pass

    # Strategy 2: Radial/Angular Perturbation Global Search
    # Bounds: Radii [0.2, 2.0], Angles [0, 2*pi]
    bounds_radial = [(0.2, 2.0)] * N + [(0, 2*np.pi)] * (N-1)
    
    def obj_radial(params):
        if time.time() > deadline - 5.0:
            return 0.0
        pts = _radial_ang_to_points(params)
        return _score_neg(pts)

    if time.time() < deadline - 20.0:
        try:
            result_radial = differential_evolution(
                obj_radial, 
                bounds_radial, 
                maxiter=50, 
                popsize=15, 
                tol=0.01, 
                seed=123,
                init='sobol'
            )
            if result_radial.fun < 0:
                pts = _radial_ang_to_points(result_radial.x)
                score = -result_radial.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    # Strategy 3: Local Refinement using Powell on Coordinates
    # This allows breaking the symmetry of the previous parameterizations
    if time.time() < deadline - 10.0:
        try:
            init_flat = best_points.flatten()
            
            def obj_flat(x):
                p = x.reshape(N, 2)
                return _score_neg(p)
                
            # Powell is good for non-smooth objectives (min of areas)
            result_local = minimize(
                obj_flat, 
                init_flat, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if result_local.fun < 0:
                pts = result_local.x.reshape(N, 2)
                score = -result_local.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
            
    # Final check: Ensure we return a valid array
    return best_points
