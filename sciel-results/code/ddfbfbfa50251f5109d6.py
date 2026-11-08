import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    # Base angles for the 13 points
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Fourier modes to use: 1, 2, 3 (low dimensional)
    # Parameters: r0 (base radius), a1, b1, a2, b2, a3, b3
    # Total 7 parameters
    # We can also add a small center offset (cx, cy) -> 9 parameters
    # Let's stick to radial deformation from center (0,0) for simplicity and convexity preservation
    # Actually, allowing center shift is good.
    # Params: cx, cy, r0, a1, b1, a2, b2, a3, b3
    n_params = 9

    def params_to_points(params):
        cx, cy, r0, a1, b1, a2, b2, a3, b3 = params
        # r(theta) = r0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta) + a3*cos(3*theta) + b3*sin(3*theta)
        # This is a polar coordinate representation.
        # To ensure convexity, we need r(theta) > 0 and the curve to be convex.
        # For small deformations, this is usually true.
        # Let's compute points.
        r = r0 + a1 * np.cos(base_angles) + b1 * np.sin(base_angles) \
                  + a2 * np.cos(2 * base_angles) + b2 * np.sin(2 * base_angles) \
                  + a3 * np.cos(3 * base_angles) + b3 * np.sin(3 * base_angles)
        
        # Ensure r is positive
        r = np.maximum(r, 0.1)
        
        x = cx + r * np.cos(base_angles)
        y = cy + r * np.sin(base_angles)
        return np.column_stack([x, y])

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def soft_objective(params, beta=100.0):
        """
        Objective: minimize -soft_min(triangle areas) for unit hull area.
        """
        pts = params_to_points(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def hard_objective(params):
        """
        Objective: minimize -min(triangle areas) for unit hull area.
        """
        pts = params_to_points(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Differential Evolution bounds
    # cx, cy in [-0.5, 0.5]
    # r0 in [0.5, 2.0]
    # a1, b1, a2, b2, a3, b3 in [-0.5, 0.5]
    bounds = [
        (-0.5, 0.5),  # cx
        (-0.5, 0.5),  # cy
        (0.5, 2.0),   # r0
        (-0.5, 0.5),  # a1
        (-0.5, 0.5),  # b1
        (-0.5, 0.5),  # a2
        (-0.5, 0.5),  # b2
        (-0.5, 0.5),  # a3
        (-0.5, 0.5),  # b3
    ]

    best_score = -float('inf')
    best_params = None

    # Differential Evolution
    # Use a moderate population size and generations to fit in time
    # popsize=15, maxiter=50
    try:
        de_res = opt.differential_evolution(
            soft_objective,
            bounds,
            args=(100.0,),
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False,  # We will do our own polish
            init='sobol'
        )
        best_params = de_res.x
        best_score = -soft_objective(best_params, 100.0)
    except Exception as e:
        # Fallback to regular polygon if DE fails
        best_params = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_score = compute_score(params_to_points(best_params))

    # Refinement using L-BFGS-B on soft objective with higher beta
    if time.time() < start_time + time_limit - 10:
        try:
            res = opt.minimize(
                lambda x: soft_objective(x, 200.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < soft_objective(best_params, 200.0):
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # Final polish using Powell on hard objective
    if time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                hard_objective,
                best_params,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate_params = res.x
            candidate_score = -hard_objective(candidate_params)
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
        except Exception:
            pass

    # Convert best params to points
    best_points = params_to_points(best_params)
    
    # One last check: if the score is bad, try a regular polygon
    reg_poly_params = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reg_poly_points = params_to_points(reg_poly_params)
    reg_poly_score = compute_score(reg_poly_points)
    
    if reg_poly_score > best_score:
        best_points = reg_poly_points
        best_score = reg_poly_score

    return best_points
