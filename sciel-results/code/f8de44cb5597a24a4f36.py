"""Heilbronn 11 triangles: Barycentric soft-min optimization with DE exploration."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Point = u*(0,0) + v*(1,0) + (1-u-v)*(0.5, S3H)
    x = v + 0.5*(1-u-v) = v + 0.5 - 0.5u - 0.5v = 0.5 - 0.5u + 0.5v
    y = S3H*(1-u-v)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = S3H * (1.0 - u - v)
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian to barycentric coordinates.
    y = S3H*(1-u-v) => 1-u-v = y/S3H => u+v = 1 - y/S3H
    x = 0.5 - 0.5u + 0.5v => 2x - 1 = -u + v => v - u = 2x - 1
    So: v = (2x - 1 + 1 - y/S3H)/2 = (2x - y/S3H)/2 = x - y/(2*S3H)
        u = 1 - y/S3H - v = 1 - y/S3H - x + y/(2*S3H) = 1 - x - y/S3H + y/(2*S3H)
        u = 1 - x - y/(2*S3H)
    """
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / (2.0 * S3H)
    v = x - y / (2.0 * S3H)
    return np.column_stack([u, v])


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


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _bary_objective(flat_params, beta):
    """Objective in barycentric space: negative soft-min area.
    Barycentric coords (u,v) with u>=0, v>=0, u+v<=1 ensure points are in triangle.
    We use bounds [0,1] for both u and v, and the constraint u+v<=1 is handled
    by the optimizer (L-BFGS-B can't handle linear inequality directly, but we
    clip and the objective naturally penalizes near-boundary degeneracy).
    """
    bary = flat_params.reshape(-1, 2)
    # Clip to valid barycentric region: u>=0, v>=0, u+v<=1
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    # Ensure u+v <= 1
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    bary = np.column_stack([u, v])
    pts = _bary_to_cart(bary)
    return -_soft_min_area(pts, beta)


def _bary_hard_objective(flat_params):
    """Hard objective for Nelder-Mead in barycentric space."""
    bary = flat_params.reshape(-1, 2)
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    bary = np.column_stack([u, v])
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)


def _penalized_hard_min(flat_params):
    """Objective for DE: negative hard-min area with penalty for out-of-triangle."""
    points = flat_params.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    ma = _min_triangle_area(points)
    return -ma + penalty


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

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

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, S3H/6],
        [5/6, S3H/6],
        [1/3, 2*S3H/3],
        [2/3, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # Convert to barycentric for optimization
    init_bary = _cart_to_bary(init)

    # ---- Phase A: Differential Evolution for global exploration ----
    if _elapsed_ok(60):
        try:
            bounds = [(0, 1), (0, S3H)] * N
            de_result = differential_evolution(
                _penalized_hard_min,
                bounds,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            de_pts = de_result.x.reshape(-1, 2)
            de_pts = _project_to_triangle(de_pts)
            _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Multi-start barycentric soft-min optimization ----
    # Build diverse starting points in barycentric space
    bases_bary = [init_bary.copy()]
    for _ in range(8):
        noise = rng.normal(0, 0.2, size=(N, 2))
        b = init_bary + noise
        b = np.clip(b, 0, 1)
        s = b[:, 0] + b[:, 1]
        mask = s > 1
        if np.any(mask):
            b[mask, 0] /= s[mask]
            b[mask, 1] /= s[mask]
        bases_bary.append(b)
    if best_pts is not None:
        bases_bary.append(_cart_to_bary(best_pts))

    n_starts = 16
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases_bary[s % len(bases_bary)]
        bary = base + rng.normal(0, 0.05, size=(N, 2))
        bary = np.clip(bary, 0, 1)
        ss = bary[:, 0] + bary[:, 1]
        mask = ss > 1
        if np.any(mask):
            bary[mask, 0] /= ss[mask]
            bary[mask, 1] /= ss[mask]

        for beta in [15.0, 40.0, 100.0, 300.0, 800.0, 2000.0]:
            if not _elapsed_ok(35):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _bary_objective(x, b),
                    bary.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                bary = res.x.reshape(-1, 2)
                pts = _bary_to_cart(bary)
                pts = _project_to_triangle(pts)
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase C: Nelder-Mead local polish in barycentric space ----
    if best_pts is not None and _elapsed_ok(15):
        best_bary = _cart_to_bary(best_pts)
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(4):
                if not _elapsed_ok(8):
                    break
                cand_bary = best_bary + rng.normal(0, scale, size=(N, 2))
                cand_bary = np.clip(cand_bary, 0, 1)
                ss = cand_bary[:, 0] + cand_bary[:, 1]
                mask = ss > 1
                if np.any(mask):
                    cand_bary[mask, 0] /= ss[mask]
                    cand_bary[mask, 1] /= ss[mask]
                try:
                    res = minimize(
                        _bary_hard_objective,
                        cand_bary.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    p_bary = res.x.reshape(-1, 2)
                    p = _bary_to_cart(p_bary)
                    p = _project_to_triangle(p)
                    _update_best(p)
                    if p is not None and _min_triangle_area(p) > best_area:
                        best_bary = _cart_to_bary(p)
                except Exception:
                    pass

    # ---- Phase D: Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(5):
        best_bary = _cart_to_bary(best_pts)
        try:
            res = minimize(
                _bary_hard_objective,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p_bary = res.x.reshape(-1, 2)
            p = _bary_to_cart(p_bary)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
