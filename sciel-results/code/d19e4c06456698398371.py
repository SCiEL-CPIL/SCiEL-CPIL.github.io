import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
_HALF_N = N // 2  # 7

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Pre-compute all triangle index combinations for vectorized area computation
    tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)

    # Base angles for 7 points in upper half-plane
    base_angles = np.linspace(0, 2 * np.pi, _HALF_N, endpoint=False)

    def _min_triangle_area_vec(points):
        """Vectorized minimum triangle area for N points."""
        x = points[:, 0]
        y = points[:, 1]
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def _hull_area_fast(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def _score_points(points):
        """Compute min triangle area / convex hull area."""
        if points.shape[0] != N:
            return 0.0
        if np.any(~np.isfinite(points)):
            return 0.0
        hull_area = _hull_area_fast(points)
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area_vec(points)
        return min_area / hull_area

    def _sym_params_to_points(params):
        """
        Convert centrally-symmetric parameter vector to 14 points.
        params: [a0, a1, ..., a6, r0, r1, ..., r6] (14 values)
        Points i (0..6) are at angle a_i, radius r_i.
        Points i+7 (7..13) are at angle a_i + pi, radius r_i.
        """
        angles = params[:_HALF_N]
        radii = np.maximum(params[_HALF_N:], 0.1)
        x_half = radii * np.cos(angles)
        y_half = radii * np.sin(angles)
        x_full = np.concatenate([x_half, -x_half])
        y_full = np.concatenate([y_half, -y_half])
        return np.column_stack([x_full, y_full])

    def _neg_score_sym(params):
        """Negative score for centrally symmetric parameterization."""
        points = _sym_params_to_points(params)
        if np.any(~np.isfinite(points)):
            return 1e10
        score = _score_points(points)
        return -score

    # Soft-min objective for smoother optimization
    def _neg_score_sym_soft(params, beta=200.0):
        """Smooth soft-min objective for centrally symmetric parameters."""
        points = _sym_params_to_points(params)
        if np.any(~np.isfinite(points)):
            return 1e10
        x = points[:, 0]
        y = points[:, 1]
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        if np.min(areas) < 1e-12:
            return 1e10
        
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        
        hull_area = _hull_area_fast(points)
        if hull_area < 1e-10:
            return 1e10
        return hull_area / soft_min_val

    # Generate diverse initial configurations
    def _generate_inits():
        inits = []
        rng = np.random.default_rng(42)
        
        # Regular 14-gon (all equal angles and radii)
        inits.append(np.concatenate([base_angles, np.ones(_HALF_N)]))
        
        # Rotated regular
        for rot in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7]:
            a = base_angles + rot
            inits.append(np.concatenate([a, np.ones(_HALF_N)]))
        
        # Non-uniform angles with uniform radii
        for seed in range(15):
            rng = np.random.default_rng(seed * 11 + 5)
            angles = np.sort(rng.uniform(0, 2 * np.pi, _HALF_N))
            inits.append(np.concatenate([angles, np.ones(_HALF_N)]))
        
        # Jittered radii
        for js in [0.05, 0.1, 0.15, 0.2]:
            rng = np.random.default_rng(int(js * 100))
            radii = 1.0 + rng.normal(0, js, _HALF_N)
            radii = np.maximum(radii, 0.5)
            inits.append(np.concatenate([base_angles, radii]))
        
        # Alternating radius patterns
        for r_pat in [
            [1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0],
            [1.0, 0.75, 0.9, 0.75, 1.0, 0.75, 0.9],
            [1.1, 0.9, 1.0, 0.9, 1.1, 0.9, 1.0],
        ]:
            inits.append(np.concatenate([base_angles, np.array(r_pat)]))
        
        # Fully random
        for seed in range(10):
            rng = np.random.default_rng(seed * 7 + 3)
            angles = np.sort(rng.uniform(0, 2 * np.pi, _HALF_N))
            radii = 1.0 + rng.normal(0, 0.2, _HALF_N)
            radii = np.maximum(radii, 0.5)
            inits.append(np.concatenate([angles, radii]))
        
        return inits

    # Main optimization loop
    best_score_val = 0.0
    best_params = None

    inits = _generate_inits()
    
    # Phase 1: Nelder-Mead on each initial guess (hard objective)
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_neg_score_sym, x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < 0:
                score_val = -res.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Refine best solution with soft-min objective (smoother landscape)
    if best_params is not None and time.time() < deadline - 15:
        try:
            res2 = minimize(_neg_score_sym_soft, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            # Re-evaluate with hard objective
            points = _sym_params_to_points(res2.x)
            score_check = _score_points(points)
            if score_check > best_score_val:
                best_score_val = score_check
                best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res3 = minimize(_neg_score_sym, best_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res3.fun < 0:
                score_val = -res3.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_params = res3.x.copy()
        except Exception:
            pass

    # Fallback: if symmetric approach failed, use regular polygon
    if best_params is None:
        best_params = np.concatenate([base_angles, np.ones(_HALF_N)])

    # Final result
    points = _sym_params_to_points(best_params)
    
    # Ensure valid output
    if not np.all(np.isfinite(points)) or points.shape != (N, 2):
        # Fallback to regular 14-gon
        angles_full = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles_full), np.sin(angles_full)])

    return points
