import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

# Fixed angles for the star-shaped parameterization
_ANGLES = 2 * np.pi * np.arange(N) / N
_COS_A = np.cos(_ANGLES)
_SIN_A = np.sin(_ANGLES)

def _radii_to_points(radii):
    """Convert radii to 2D points."""
    x = radii * _COS_A
    y = radii * _SIN_A
    return np.column_stack([x, y])

def _polygon_area(points):
    """Calculate area of the polygon formed by points in order."""
    x = points[:, 0]
    y = points[:, 1]
    # Shoelace formula
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return 0.5 * np.abs(np.sum(x * y_next - x_next * y))

def _get_triangle_areas(points):
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    # For the star-shaped configuration, the hull is the polygon itself
    # But to be safe, we can use ConvexHull if we do full 2D polish later.
    # For the 13-param optimization, we assume convex position.
    hull_area = _polygon_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score_params(radii, alpha):
    """
    Objective function for optimization.
    radii: (13,) array
    alpha: temperature for softmin
    Returns: negative score to be minimized.
    """
    points = _radii_to_points(radii)
    hull_area = _polygon_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    
    # Softmin of areas
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return -(softmin_val / hull_area)

def _neg_score_params(radii):
    """Exact objective for final polish in 13D space."""
    points = _radii_to_points(radii)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Regular polygon (all radii = 1)
    r0 = np.ones(N)
    
    # Stage 1: Optimize radii with smooth objective
    # We will try a few different alphas (temperatures)
    best_r = r0.copy()
    best_score = _score(_radii_to_points(best_r))
    
    # Multi-stage softmin optimization
    for alpha in [50, 20, 10, 5, 2]:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                _softmin_score_params,
                best_r,
                args=(alpha,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                cand_r = res.x
                cand_score = _score(_radii_to_points(cand_r))
                if cand_score > best_score:
                    best_score = cand_score
                    best_r = cand_r
        except Exception:
            pass

    # Stage 2: Nelder-Mead on exact objective in 13D
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _neg_score_params,
                best_r,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_r = res.x
            cand_score = _score(_radii_to_points(cand_r))
            if cand_score > best_score:
                best_score = cand_score
                best_r = cand_r
        except Exception:
            pass
            
    # Convert best radii to points
    best_points = _radii_to_points(best_r)
    
    # Stage 3: Full 2D Polish
    # The optimal solution might not be perfectly star-shaped with fixed angles.
    # We allow the points to move freely in 2D.
    
    def _score_2d(points):
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area < 1e-10:
            return 0.0
        areas = _get_triangle_areas(points)
        min_area = np.min(areas)
        return min_area / hull_area

    def _neg_score_2d(params):
        points = params.reshape(N, 2)
        return -_score_2d(points)
        
    # Initialize 2D optimization with the 13D result
    current_pts = best_points.copy()
    current_score = _score_2d(current_pts)
    
    # Use Nelder-Mead on the 2D space
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score_2d,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score_2d(cand_pts)
            if cand_score > current_score:
                current_score = cand_score
                current_pts = cand_pts
        except Exception:
            pass
            
    # Stage 4: Coordinate Descent Polish in 2D
    # This is very effective for max-min problems.
    # For each point, try small moves in x and y.
    
    def _polish(points, deadline):
        pts = points.copy()
        score = _score_2d(pts)
        
        # Adaptive step size
        step = 0.05
        
        for iter in range(100):
            if time.time() > deadline - 2:
                break
                
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                orig = pts[i].copy()
                best_local = orig
                best_local_score = score
                
                # Try a few random perturbations
                for _ in range(20):
                    dx = np.random.uniform(-step, step)
                    dy = np.random.uniform(-step, step)
                    cand = pts.copy()
                    cand[i] = orig + np.array([dx, dy])
                    s = _score_2d(cand)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_local = orig + np.array([dx, dy])
                        
                if not np.allclose(best_local, orig):
                    pts[i] = best_local
                    score = best_local_score
                    improved = True
                    
            if not improved:
                # Reduce step size if no improvement
                step *= 0.5
                if step < 1e-6:
                    break
                    
        return pts, score

    if time.time() < deadline - 5:
        polished_pts, polished_score = _polish(current_pts, deadline)
        if polished_score > current_score:
            current_score = polished_score
            current_pts = polished_pts

    return current_pts
