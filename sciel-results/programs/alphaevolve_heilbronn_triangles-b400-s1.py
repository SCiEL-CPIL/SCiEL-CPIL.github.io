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

DEADLINE = 140  # safety margin from 150s limit

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

# Barycentric coordinate conversion matrices
# Point = a*V0 + b*V1 + c*V2 where a+b+c=1, a,b,c >= 0
# x = b + c*0.5, y = c*S3H
# So: b = x - 0.5*c, c = y/S3H, a = 1 - b - c
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

def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _barycentric_penalty(abc):
    """Penalty for barycentric coords outside [0,1] or not summing to 1."""
    # Each coordinate should be in [0,1]
    penalty = 0.0
    for i in range(abc.shape[0]):
        for j in range(3):
            if abc[i, j] < 0:
                penalty += abc[i, j] ** 2 * 100
            elif abc[i, j] > 1:
                penalty += (abc[i, j] - 1) ** 2 * 100
    # Sum should be 1
    sums = abc.sum(axis=1)
    penalty += np.sum((sums - 1.0) ** 2) * 100
    return penalty

def _objective_bary(flat, beta):
    """Objective in barycentric space: minimize -soft_min + penalty."""
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _soft_min_area(pts, beta)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _objective_bary_hard(flat):
    """Hard min objective in barycentric space."""
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _min_triangle_area(pts)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _push_apart(pts, alpha=1.0):
    """Gradient-free push: move each point away from others to increase min area."""
    n = len(pts)
    displacement = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                # Push apart proportionally to 1/dist^2
                force = alpha / (dist * dist)
                displacement[i] += d / dist * force
                displacement[j] -= d / dist * force
    return pts + displacement

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

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        a = 1 - u - v
        x = a * 0 + u * 1 + v * 0.5
        y = a * 0 + u * 0 + v * S3H
        return np.array([x, y])

    def _random_barycentric():
        """Generate random point in barycentric coords (a,b,c) with a+b+c=1, all >= 0."""
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        a = 1 - r1 - r2
        b = r1
        c = r2
        return np.array([a, b, c])

    # ---- Phase A: Differential Evolution in barycentric space ----
    if _elapsed_ok(50):
        try:
            # Barycentric bounds: each coord in [0, 1], but sum must be 1
            # We use a relaxed approach: optimize in 2D barycentric (b, c) where a = 1-b-c
            # b >= 0, c >= 0, b + c <= 1
            def _de_obj_2d(flat_bc):
                """DE objective using 2D barycentric (b, c) pairs."""
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                # Penalty for constraint violation
                pen = 0.0
                for i in range(N):
                    if a[i] < 0:
                        pen += a[i] ** 2 * 500
                    if b[i] < 0:
                        pen += b[i] ** 2 * 500
                    if c[i] < 0:
                        pen += c[i] ** 2 * 500
                    if a[i] > 1:
                        pen += (a[i] - 1) ** 2 * 500
                    if b[i] > 1:
                        pen += (b[i] - 1) ** 2 * 500
                    if c[i] > 1:
                        pen += (c[i] - 1) ** 2 * 500
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma + pen

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj_2d,
                bounds_de,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-5,
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
                # Clip to valid region
                for i in range(N):
                    for j in range(3):
                        abc[i, j] = max(0.0, min(1.0, abc[i, j]))
                    s = abc[i].sum()
                    if s > 0:
                        abc[i] /= s
                de_pts = _barycentric_to_cartesian(abc)
                de_pts = _project_to_triangle(de_pts)
                _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Barycentric soft-min gradient optimization ----
    n_starts = 20
    bases_cart = [init.copy()]
    for _ in range(12):
        bases_cart.append(init + rng.normal(0, 0.25, size=(N, 2)))
    if best_pts is not None:
        bases_cart.append(best_pts.copy())

    for s in range(n_starts):
        if not _elapsed_ok(35):
            break
        base = bases_cart[s % len(bases_cart)]
        pts = base + rng.normal(0, 0.05, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        # Convert to barycentric for optimization
        abc = _cartesian_to_barycentric(pts)
        # Clip to valid barycentric
        for i in range(N):
            for j in range(3):
                abc[i, j] = max(0.0, min(1.0, abc[i, j]))
            s_val = abc[i].sum()
            if s_val > 0:
                abc[i] /= s_val

        for beta in [15.0, 40.0, 100.0, 250.0, 600.0, 1200.0]:
            if not _elapsed_ok(30):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_bary(x, b),
                    abc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                abc_new = res.x.reshape(-1, 3)
                # Normalize to valid barycentric
                for i in range(N):
                    for j in range(3):
                        abc_new[i, j] = max(0.0, min(1.0, abc_new[i, j]))
                    s_val = abc_new[i].sum()
                    if s_val > 0:
                        abc_new[i] /= s_val
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc = abc_new
            except Exception:
                pass

    # ---- Phase C: Push-apart + Nelder-Mead hybrid local search ----
    if best_pts is not None and _elapsed_ok(20):
        for iteration in range(30):
            if not _elapsed_ok(10):
                break
            # Push apart step
            cand = best_pts.copy()
            for _ in range(3):
                cand = _push_apart(cand, alpha=0.005)
                cand = _project_to_triangle(cand)
            _update_best(cand)

            # Nelder-Mead polish
            try:
                res = minimize(
                    _objective_bary_hard,
                    _cartesian_to_barycentric(best_pts).flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                abc_res = res.x.reshape(-1, 3)
                for i in range(N):
                    for j in range(3):
                        abc_res[i, j] = max(0.0, min(1.0, abc_res[i, j]))
                    s_val = abc_res[i].sum()
                    if s_val > 0:
                        abc_res[i] /= s_val
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    # ---- Phase D: Final fine polish ----
    if best_pts is not None and _elapsed_ok(8):
        for _ in range(3):
            if not _elapsed_ok(4):
                break
            try:
                abc = _cartesian_to_barycentric(best_pts)
                for i in range(N):
                    for j in range(3):
                        abc[i, j] = max(0.0, min(1.0, abc[i, j]))
                    s_val = abc[i].sum()
                    if s_val > 0:
                        abc[i] /= s_val
                res = minimize(
                    _objective_bary_hard,
                    abc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                abc_res = res.x.reshape(-1, 3)
                for i in range(N):
                    for j in range(3):
                        abc_res[i, j] = max(0.0, min(1.0, abc_res[i, j]))
                    s_val = abc_res[i].sum()
                    if s_val > 0:
                        abc_res[i] /= s_val
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
