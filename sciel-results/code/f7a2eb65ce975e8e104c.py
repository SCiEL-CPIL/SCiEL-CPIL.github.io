import numpy as np
from scipy.optimize import minimize
import time

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute triangle indices for fast scoring
    tri_idx = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_idx.append((i, j, k))
    tri_idx = np.array(tri_idx, dtype=np.int32)
    I = tri_idx[:, 0]
    J = tri_idx[:, 1]
    K = tri_idx[:, 2]

    def _score(pts):
        """Compute min triangle area / convex hull area."""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-12:
                return 0.0
        except:
            return 0.0

        p_i = pts[I]
        p_j = pts[J]
        p_k = pts[K]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        min_area = 0.5 * np.abs(cross).min()
        return min_area / hull_area

    def _neg_score(params):
        """Negative score for minimization. params = [r_hex, r_hept, phase_hex, phase_hept]"""
        r_hex = params[0]
        r_hept = params[1]
        phase_hex = params[2]
        phase_hept = params[3]

        # 6-point regular hexagon
        angles_hex = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phase_hex
        pts_hex = r_hex * np.column_stack((np.cos(angles_hex), np.sin(angles_hex)))

        # 7-point regular heptagon
        angles_hept = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phase_hept
        pts_hept = r_hept * np.column_stack((np.cos(angles_hept), np.sin(angles_hept)))

        pts = np.vstack([pts_hex, pts_hept])
        return -_score(pts)

    # Strategy: Grid search + local optimization over 4 parameters
    best_score = 0.0
    best_params = None

    # Coarse grid search
    r_hex_range = np.linspace(0.3, 1.0, 10)
    r_hept_range = np.linspace(0.3, 1.0, 10)
    phase_hex_range = np.linspace(0, np.pi / 3, 5)
    phase_hept_range = np.linspace(0, 2 * np.pi / 7, 5)

    for r_h in r_hex_range:
        for r_p in r_hept_range:
            for ph_h in phase_hex_range:
                for ph_p in phase_hept_range:
                    if time.time() > deadline:
                        break
                    params = [r_h, r_p, ph_h, ph_p]
                    s = -_neg_score(params)
                    if s > best_score:
                        best_score = s
                        best_params = params
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Local optimization from best grid point
    if best_params is not None:
        try:
            res = minimize(_neg_score, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500})
            if res.fun < 0:
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_params = res.x
        except Exception:
            pass

    # Try a few random starts for local optimization
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        r0 = [rng.uniform(0.4, 0.9), rng.uniform(0.4, 0.9),
              rng.uniform(0, np.pi / 3), rng.uniform(0, 2 * np.pi / 7)]
        try:
            res = minimize(_neg_score, r0, method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 300})
            if res.fun < 0:
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_params = res.x
        except Exception:
            pass

    # Generate final points from best parameters
    if best_params is not None:
        r_hex, r_hept, phase_hex, phase_hept = best_params
        angles_hex = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phase_hex
        pts_hex = r_hex * np.column_stack((np.cos(angles_hex), np.sin(angles_hex)))
        angles_hept = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phase_hept
        pts_hept = r_hept * np.column_stack((np.cos(angles_hept), np.sin(angles_hept)))
        return np.vstack([pts_hex, pts_hept])
    else:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
