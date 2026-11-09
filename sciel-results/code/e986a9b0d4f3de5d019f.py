import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Precompute triple indices for N=14
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

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0

    return min_area / hull_area

def score_from_angles(angles):
    """
    Calculate score for points on unit circle at given sorted angles.
    angles: np.array of shape (N,)
    """
    # Ensure angles are sorted and unique enough
    # We assume the input angles are sorted and the first is 0.
    # We add 2*pi to the last to close the loop for hull calculation if needed, 
    # but ConvexHull handles the points directly.
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # For points on the unit circle, the convex hull area is simply the area of the polygon.
    # We can calculate it faster without ConvexHull if we know they are in convex position.
    # Area of polygon = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    # Since they are on unit circle and sorted by angle, they are in convex position.
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    hull_area = 0.5 * np.sum(x * y_next - x_next * y)
    
    if hull_area <= 1e-10:
        return 0.0

    # Min Triangle Area
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0

    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 140s to be safe.
    deadline = start_time + 140.0

    best_score = 0.0
    best_angles = None

    # 1. Regular Polygon Baseline
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_score = score_from_angles(base_angles)
    best_score = base_score
    best_angles = base_angles.copy()

    # 2. Local Optimization on Angles
    # We optimize the angles to maximize the score.
    # We fix the first angle to 0 to remove rotational symmetry.
    # We optimize the remaining N-1 angles.
    # We must ensure they remain sorted: 0 < a_1 < ... < a_{N-1} < 2*pi.
    # To handle the sorting constraint, we can optimize the *differences* (gaps) between angles.
    # Let g_i be the gap between angle i and i+1. Sum(g_i) = 2*pi.
    # This is a constrained optimization. We can use a simple grid search on the gaps
    # or use an unconstrained optimizer with a penalty for unsorted angles.
    # Given the low dimensionality (13 variables), a simple local search or Nelder-Mead
    # on the angles with a sorting check is efficient.

    def objective_angles(flat_angles):
        # flat_angles contains N-1 angles (indices 1 to N-1)
        a = np.zeros(N)
        a[0] = 0.0
        a[1:] = flat_angles
        
        # Check if sorted and within [0, 2*pi)
        # If not, return a large penalty
        if np.any(a[1:] < 0) or np.any(a[1:] >= 2 * np.pi):
            return -1.0 # Minimize -score, so large negative score is bad
        # Check strict increasing order
        if np.any(np.diff(a) <= 1e-9):
            return -1.0
            
        return -score_from_angles(a)

    # Initial guess for the optimizer: regular polygon angles
    init_flat = base_angles[1:].copy()

    # Use Nelder-Mead for robustness on this small problem
    try:
        res = minimize(
            objective_angles, 
            init_flat, 
            method='Nelder-Mead', 
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 5000}
        )
        
        # Validate result
        res_angles = np.zeros(N)
        res_angles[0] = 0.0
        res_angles[1:] = res.x
        
        # Ensure validity
        if np.all(np.diff(res_angles) > 1e-9) and res_angles[-1] < 2 * np.pi:
            res_score = -res.fun
            if res_score > best_score:
                best_score = res_score
                best_angles = res_angles
    except Exception:
        pass

    # 3. Grid Search for Symmetry Breaking
    # Sometimes a slight perturbation of the regular polygon yields a better minimum triangle area.
    # We try a few structured perturbations.
    
    # Perturbation 1: Alternating radius? No, we are restricted to unit circle in this block.
    # The current strategy is unit circle. The previous program allowed radial variation.
    # Let's add a quick check for radial variation using the original get_score.
    
    # We will run a few short Powell optimizations on the full 2D coordinates,
    # but starting from the regular polygon and its angular-optimized version.
    
    candidates = [base_angles, best_angles]
    
    for ang_candidate in candidates:
        x = np.cos(ang_candidate)
        y = np.sin(ang_candidate)
        init_pts = np.column_stack((x, y))
        
        # Quick Powell on 2D to allow radial changes
        def objective_2d(flat_pts):
            pts = flat_pts.reshape(N, 2)
            return -get_score(pts)
        
        try:
            res_2d = minimize(
                objective_2d,
                init_pts.flatten(),
                method='Powell',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 2000}
            )
            
            if time.time() > deadline - 10:
                break
                
            if res_2d.fun < -best_score:
                new_score = -res_2d.fun
                if new_score > best_score:
                    best_score = new_score
                    # We need to return points, so we store the best points
                    # For now, we just track score. We will reconstruct at the end.
                    # We need to keep track of the best points array.
                    pass # We'll handle point reconstruction below
        except Exception:
            pass

    # To properly combine, we need to track the best points array.
    # Let's restart the tracking logic to be cleaner.
    
    best_points = None
    best_score_final = 0.0
    
    # Regular Polygon
    reg_pts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    s_reg = get_score(reg_pts)
    if s_reg > best_score_final:
        best_score_final = s_reg
        best_points = reg_pts

    # Optimized Angles (Unit Circle)
    if best_angles is not None:
        ang_pts = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
        s_ang = get_score(ang_pts)
        if s_ang > best_score_final:
            best_score_final = s_ang
            best_points = ang_pts

    # 2D Refinement on best candidates
    candidates_pts = [reg_pts]
    if best_angles is not None and not np.allclose(reg_pts, ang_pts):
        candidates_pts.append(ang_pts)
        
    for init_pts in candidates_pts:
        if time.time() > deadline - 15:
            break
            
        def obj_2d(flat_pts):
            pts = flat_pts.reshape(N, 2)
            return -get_score(pts)
        
        try:
            # Use a more robust method with tighter tolerances
            res = minimize(
                obj_2d, 
                init_pts.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 2000}
            )
            
            if time.time() > deadline - 5:
                break
                
            if res.fun < -best_score_final:
                new_score = -res.fun
                if new_score > best_score_final:
                    best_score_final = new_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            pass

    # Final Fallback
    if best_points is None:
        best_points = reg_pts

    return best_points
