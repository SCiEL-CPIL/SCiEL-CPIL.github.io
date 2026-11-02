import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import sys

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    # Precompute all triangle indices
    tri_indices = list(combinations(range(N), 3))

    def compute_score(pts):
        """Compute min triangle area / hull area for a set of points."""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
            if hull_area < 1e-12:
                return 0.0
        except Exception:
            return 0.0

        # Compute all triangle areas vectorized
        min_area = float('inf')
        for i, j, k in tri_indices:
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            area = 0.5 * abs(cross)
            if area < min_area:
                min_area = area

        return min_area / hull_area

    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        # Check for degeneracy
        if np.allclose(pts, pts[0]):
            return 0.0
        score = compute_score(pts)
        return -score

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()

    best_score_val = 0
    best_result = x0.copy()

    # First, evaluate the initial guess
    initial_score = compute_score(x0.reshape(N, 2))
    best_score_val = initial_score
    best_result = x0.copy()

    # Optimization with Powell method
    try:
        res = minimize(neg_score, x0, method='Powell', 
                       options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False})
        if time.time() > deadline:
            pass
        if res.fun < 0:  # Negative means positive score
            score_val = -res.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res.x.copy()
    except Exception:
        pass

    # If time remains, try a few perturbed starts with Nelder-Mead
    if time.time() < deadline - 10:
        for pert_scale in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline - 5:
                break
            x0_pert = x0 + np.random.normal(0, pert_scale, x0.shape)
            try:
                res2 = minimize(neg_score, x0_pert, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
                if res2.fun < 0:
                    score_val = -res2.fun
                    if score_val > best_score_val:
                        best_score_val = score_val
                        best_result = res2.x.copy()
            except Exception:
                pass

    # Final polish: use the best result found and do a tight Powell refinement
    if time.time() < deadline - 5 and best_score_val > 0:
        try:
            res3 = minimize(neg_score, best_result, method='Powell',
                           options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False})
            if res3.fun < 0:
                score_val = -res3.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_result = res3.x.copy()
        except Exception:
            pass

    return best_result.reshape(N, 2)
