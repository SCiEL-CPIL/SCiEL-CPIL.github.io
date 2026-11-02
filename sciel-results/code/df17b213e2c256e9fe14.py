"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

def _cartesian_to_barycentric(pts):
    """Convert (x,y) points to (a,b,c) barycentric coords with a+b+c=1."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _barycentric_to_cartesian(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _push_min_area(pts, step_size=0.01, max_iter=100):
    """
    Directly push the points of the minimum area triangle apart to increase its area.
    This is a local search heuristic.
    """
    current_area = _min_triangle_area(pts)
    pts = pts.copy()
    
    for _ in range(max_iter):
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        
        # Get the points of the minimum triangle
        i, j, k = _TRIPLETS[min_idx]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # Calculate gradient of area with respect to pi, pj, pk
        dx_ji = pj[0] - pi[0]
        dy_ji = pj[1] - pi[1]
        dx_ki = pk[0] - pi[0]
        dy_ki = pk[1] - pi[1]
        cross = dx_ji * dy_ki - dy_ji * dx_ki
        
        if cross >= 0:
            g_pi = np.array([pj[1] - pk[1], pj[0] - pk[0]])
            g_pj = np.array([pk[1] - pi[1], -(pk[0] - pi[0])])
            g_pk = np.array([-(pj[1] - pi[1]), pj[0] - pi[0]])
        else:
            g_pi = np.array([-(pj[1] - pk[1]), -(pj[0] - pk[0])])
            g_pj = np.array([-(pk[1] - pi[1]), (pk[0] - pi[0])])
            g_pk = np.array([pj[1] - pi[1], -(pj[0] - pi[0])])
            
        # Normalize gradients
        norm_pi = np.linalg.norm(g_pi)
        norm_pj = np.linalg.norm(g_pj)
        norm_pk = np.linalg.norm(g_pk)
        
        if norm_pi > 1e-8: g_pi /= norm_pi
        if norm_pj > 1e-8: g_pj /= norm_pj
        if norm_pk > 1e-8: g_pk /= norm_pk
        
        # Update points
        pts[i] += step_size * g_pi
        pts[j] += step_size * g_pj
        pts[k] += step_size * g_pk
        
        # Project
        _project_to_triangle(pts)
        
        # Check improvement
        new_area = _min_triangle_area(pts)
        if new_area > current_area + 1e-10:
            current_area = new_area
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return pts

def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial configuration: symmetric placement ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: Differential Evolution in barycentric space (Quick) ----
    if _elapsed_ok(40):
        try:
            def _de_obj_2d(flat_bc):
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj_2d,
                bounds_de,
                maxiter=20,
                popsize=10,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_result.fun < 0:
                bc = de_result.x.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                de_pts = _barycentric_to_cartesian(abc)
                de_pts = _project_to_triangle(de_pts)
                _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Push Apart Local Search ----
    if best_pts is not None and _elapsed_ok(30):
        pushed = _push_min_area(best_pts, step_size=0.05, max_iter=200)
        _update_best(pushed)
        
        pushed_init = _push_min_area(init, step_size=0.05, max_iter=200)
        _update_best(pushed_init)
        
        for _ in range(5):
            if not _elapsed_ok(5): break
            rand_pts = rng.uniform(0, 1, size=(N, 2))
            rand_pts[:, 1] *= S3H
            rand_pts = _project_to_triangle(rand_pts)
            pushed_rand = _push_min_area(rand_pts, step_size=0.05, max_iter=100)
            _update_best(pushed_rand)

    # ---- Phase C: Soft-Min Gradient Optimization (Polishing) ----
    if best_pts is not None and _elapsed_ok(30):
        abc_best = _cartesian_to_barycentric(best_pts)
        abc_best = np.clip(abc_best, 0, 1)
        sums = abc_best.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_best = abc_best / sums

        def _soft_min_area(pts, beta):
            areas = _all_triangle_areas(pts)
            max_a = np.max(areas)
            shifted = -beta * (areas - max_a)
            log_sum = np.log(np.sum(np.exp(shifted)))
            soft_min = max_a - log_sum / beta
            return soft_min

        for beta in [100.0, 500.0, 1000.0]:
            if not _elapsed_ok(10): break
            try:
                def _obj_soft(x, b=beta):
                    abc_local = x.reshape(-1, 3)
                    abc_local = np.clip(abc_local, 0, 1)
                    sums = abc_local.sum(axis=1, keepdims=True)
                    sums[sums == 0] = 1
                    abc_local = abc_local / sums
                    pts_local = _barycentric_to_cartesian(abc_local)
                    return -_soft_min_area(pts_local, b)

                res = minimize(
                    _obj_soft,
                    abc_best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                abc_new = res.x.reshape(-1, 3)
                abc_new = np.clip(abc_new, 0, 1)
                sums = abc_new.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_new = abc_new / sums
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc_best = abc_new
            except Exception:
                pass

    if best_pts is None:
        return init

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
