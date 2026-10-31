import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13

# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

# Define the base lattice: 4x4 grid from 0..3, remove top-right corner (3,3)
# This creates a 13-point set with high regularity.
# Coordinates:
# (0,0), (1,0), (2,0), (3,0)
# (0,1), (1,1), (2,1), (3,1)
# (0,2), (1,2), (2,2), (3,2)
# (0,3), (1,3), (2,3)
_base_points = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0],
    [0, 1], [1, 1], [2, 1], [3, 1],
    [0, 2], [1, 2], [2, 2], [3, 2],
    [0, 3], [1, 3], [2, 3]
])

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _affine_points(params):
    """
    Map the base lattice points using an affine transformation.
    params: [a, b, c, d] representing matrix [[a, b], [c, d]]
    The transformation is P = P_base @ A.T + t (we can ignore translation t 
    because the score is invariant to translation and scaling is handled by 
    the ratio, but actually A handles scaling and shear. 
    Wait, the ratio min_area/hull_area is invariant to uniform scaling and translation.
    It is NOT invariant to shear or non-uniform scaling.
    So we optimize A = [[a, b], [c, d]].
    Since the ratio is scale-invariant, we can fix one parameter, e.g., d=1, 
    or just let the optimizer find the scale. Let's let it find the scale, 
    but normalize A to prevent numerical issues? 
    Actually, the objective function is scale-invariant, so the optimizer 
    might wander in scale. It's better to fix the scale.
    Let's fix the determinant of A to 1? Or just fix one element.
    Let's fix d=1. Then we optimize a, b, c.
    However, differential_evolution works better with bounded 3D search.
    Let's use 3 parameters: a, b, c with d=1.
    """
    a, b, c = params
    d = 1.0
    A = np.array([[a, b], [c, d]])
    # Apply transformation: P_new = P_base @ A.T
    return _base_points @ A.T

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    pts = _affine_points(params)
    score = calculate_score(pts)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_params = None
    best_points = None
    
    # Strategy: 
    # 1. Global search using Differential Evolution on the 3 affine parameters.
    # 2. Local polish using L-BFGS-B on the same 3 parameters.
    
    # Bounds for a, b, c. 
    # The lattice is roughly square. 
    # If we shear too much, points become collinear or close.
    # Let's search in a reasonable range, e.g., [-2, 2] for off-diagonals, 
    # and [0.5, 2] for diagonal (since d=1, a should be positive to keep orientation).
    # Actually, a can be negative (reflection), but let's keep it simple.
    # Let's allow a in [0.2, 2.0], b in [-2.0, 2.0], c in [-2.0, 2.0].
    
    bounds = [(0.2, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    # 1. Differential Evolution
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            de_result = differential_evolution(
                _objective, 
                bounds, 
                maxiter=50, 
                tol=1e-6, 
                popsize=15,
                seed=42,
                polish=False # We will polish manually
            )
            
            params_de = de_result.x
            score_de = calculate_score(_affine_points(params_de))
            
            if score_de > best_score:
                best_score = score_de
                best_params = params_de.copy()
                best_points = _affine_points(params_de)
                
        except Exception:
            pass
    
    # 2. Local Polish with L-BFGS-B
    remaining = deadline - time.time()
    if remaining > 5 and best_params is not None:
        try:
            # Start from best DE result
            res = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            params_lbfgs = res.x
            score_lbfgs = calculate_score(_affine_points(params_lbfgs))
            
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_params = params_lbfgs.copy()
                best_points = _affine_points(params_lbfgs)
                
        except Exception:
            pass
            
    # 3. Try a few random restarts of local optimization if time permits
    remaining = deadline - time.time()
    if remaining > 10:
        # Generate a few random initializations in the bounds
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            try:
                # Random start
                p0 = np.array([
                    np.random.uniform(0.5, 1.5),
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0)
                ])
                
                res = minimize(
                    _objective, 
                    p0, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                params_r = res.x
                score_r = calculate_score(_affine_points(params_r))
                
                if score_r > best_score:
                    best_score = score_r
                    best_params = params_r.copy()
                    best_points = _affine_points(params_r)
                    
            except Exception:
                continue

    # Fallback: Identity transformation (a=1, b=0, c=0)
    if best_points is None:
        best_params = np.array([1.0, 0.0, 0.0])
        best_points = _affine_points(best_params)
        best_score = calculate_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
