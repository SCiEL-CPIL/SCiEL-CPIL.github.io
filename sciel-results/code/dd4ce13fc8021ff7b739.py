import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
# Pre-compute all triangle index combinations for vectorization
# Indices are 0..13
tri_indices = np.array(list(combinations(range(N), 3)))

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: shape (14, 2)
    """
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    points: shape (14, 2)
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score is min_triangle_area / convex_hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 14 parameters (7 points) to 14 points (7 points + 7 negatives).
    params: shape (14,) -> [x0, y0, x1, y1, ..., x6, y6]
    """
    half = params.reshape(7, 2)
    # Ensure non-degenerate
    if np.allclose(half, half[0]):
        return np.zeros((N, 2))
    
    # The 14 points are half and -half
    # We interleave or just concatenate? 
    # The set of points is {p1..p7, -p1..-p7}.
    # The order doesn't matter for the score, but let's be consistent.
    pts = np.vstack([half, -half])
    return pts

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    pts = _params_to_points(params)
    # Check for NaNs or Infs
    if not np.all(np.isfinite(pts)):
        return 1e10
    score = _score(pts)
    return -score

def _generate_initial_configs():
    """
    Generate diverse initial configurations for the 7 independent points.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular heptagon (symmetric)
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    # Scale doesn't matter for score, but keep it reasonable
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # 2. Elliptical shapes
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.5, 0.6), (0.6, 1.5)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    # 3. Perturbed regular heptagon
    for js in [0.1, 0.2, 0.3, 0.5]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
        
    # 4. Random convex-ish polygons (7 points)
    for seed in range(30):
        rs = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.5, 1.5, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_params, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            # Reverse map to params if possible, or just store points
            # Since we return points, we just store points.
            # But for further optimization, we need params.
            # We can try to recover params from points, but it's ambiguous.
            # Better to keep the params that generated the best points.
            # However, update_best is called with points.
            # Let's modify the flow to pass params as well.
            
    # Phase 1: Multi-start Nelder-Mead on symmetric parameters
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    for init in inits:
        if time.time() > deadline:
            break
        pts = _params_to_points(init)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_params = init.copy()
            
    # Run NM on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < 0:
                score = -res.fun
                pts = _params_to_points(res.x)
                actual_score = _score(pts)
                if actual_score > best_score:
                    best_score = actual_score
                    best_points = pts.copy()
                    best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Differential Evolution for global exploration
    if time.time() < deadline - 30:
        # Bounds for parameters: [-2, 2] is usually sufficient
        bounds = [(-2.0, 2.0)] * 14
        
        try:
            # Use a callback to check time? DE doesn't have a direct time callback in older scipy, 
            # but we can just limit iterations.
            # 14 dims, popsize 10 -> 140 evals per gen. 
            # If we do 50 gens, that's 7000 evals.
            # Each eval is fast (~1ms). 7s total. Safe.
            
            result = differential_evolution(
                _objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6, 
                seed=42,
                polish=False,
                update_interval=5
            )
            
            if result.fun < 0:
                score = -result.fun
                pts = _params_to_points(result.x)
                actual_score = _score(pts)
                if actual_score > best_score:
                    best_score = actual_score
                    best_points = pts.copy()
                    best_params = result.x.copy()
                    
                    # Polish the DE result
                    if time.time() < deadline - 10:
                        res_polish = minimize(
                            _objective, 
                            result.x, 
                            method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                        )
                        if res_polish.fun < 0:
                            score2 = -res_polish.fun
                            pts2 = _params_to_points(res_polish.x)
                            actual_score2 = _score(pts2)
                            if actual_score2 > best_score:
                                best_score = actual_score2
                                best_points = pts2.copy()
                                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish with High Precision
    if best_params is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_final.fun < 0:
                score_final = -res_final.fun
                pts_final = _params_to_points(res_final.x)
                actual_score_final = _score(pts_final)
                if actual_score_final > best_score:
                    best_score = actual_score_final
                    best_points = pts_final.copy()
        except Exception:
            pass

    # Fallback: Regular 14-gon (which is centrally symmetric)
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points

    return best_points
