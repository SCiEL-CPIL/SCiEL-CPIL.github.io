"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRI = len(_TRI_IDX)


def _bary_to_cart(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    return bary[:, 1] * V1 + bary[:, 2] * V2


def _cart_to_bary(pts):
    """Convert (N,2) cartesian to (N,3) barycentric (unclamped)."""
    b = pts[:, 0] - pts[:, 1] / np.sqrt(3)
    c = 2 * pts[:, 1] / np.sqrt(3)
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _clamp_bary(bary):
    """Clamp barycentric coords to the triangle (a,b,c >= 0, sum=1)."""
    a = np.maximum(0.0, bary[:, 0])
    b = np.maximum(0.0, bary[:, 1])
    c = np.maximum(0.0, bary[:, 2])
    s = a + b + c
    s = np.maximum(s, 1e-12)
    return np.column_stack([a / s, b / s, c / s])


def _areas_bary(bary):
    """All triangle areas given barycentric coords. Vectorized."""
    pi = bary[_TRI_IDX[:, 0]]
    pj = bary[_TRI_IDX[:, 1]]
    pk = bary[_TRI_IDX[:, 2]]
    # Cross product in barycentric: |a_i(b_j c_k - c_j b_k) + b_i(c_j a_k - a_j c_k) + c_i(a_j b_k - b_j a_k)|
    cross = (pi[:, 0] * (pj[:, 1] * pk[:, 2] - pj[:, 2] * pk[:, 1])
             + pi[:, 1] * (pj[:, 2] * pk[:, 0] - pj[:, 0] * pk[:, 2])
             + pi[:, 2] * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
    return 0.5 * np.abs(cross)


def _min_area_bary(bary):
    return float(np.min(_areas_bary(bary)))


def _soft_min_bary(bary, beta=60.0):
    areas = _areas_bary(bary)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta


def _hexagonal_init():
    """Structured 3-fold symmetric init in barycentric coords."""
    bary = []
    # 3 near-vertices
    m = 0.12
    bary.append([1 - m, m / 2, m / 2])
    bary.append([m / 2, 1 - m, m / 2])
    bary.append([m / 2, m / 2, 1 - m])
    # 3 near-edge-midpoints
    m2 = 0.14
    bary.append([0.5 * (1 - m2), 0.5 * (1 - m2), m2])
    bary.append([m2, 0.5 * (1 - m2), 0.5 * (1 - m2)])
    bary.append([0.5 * (1 - m2), m2, 0.5 * (1 - m2)])
    # center
    bary.append([1 / 3, 1 / 3, 1 / 3])
    # 4 ring around center in barycentric
    cx, cy = 1 / 3, 1 / 3
    r = 0.16
    for ang in [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]:
        bary.append([cx + r * np.cos(ang), cy + r * np.sin(ang), 1 - cx - cy - r * np.cos(ang)])
    b = np.array(bary, dtype=float)
    return _clamp_bary(b)


def _random_bary_init(seed):
    rng = np.random.default_rng(seed)
    bary = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random() * (1 - r1)
        bary.append([1 - r1 - r2, r1, r2])
    return np.array(bary)


def _slsqp_max_alpha(bary0, alpha0, deadline_fn):
    """Maximize alpha s.t. all triangle areas >= alpha, barycentric feasibility."""
    n = N * 3

    def unpack(x):
        b = x[:n].reshape(N, 3)
        a = x[n]
        return b, a

    def objective(x):
        b, a = unpack(x)
        b = _clamp_bary(b)
        # Penalty for area violations + maximize alpha
        areas = _areas_bary(b)
        viol = np.maximum(0.0, a - areas)
        # Soft: use a log-barrier-like penalty
        pen = np.sum(viol ** 2) * 100.0
        return -(a) + pen

    def jac(x):
        # Numerical gradient (finite difference) - acceptable for SLSQP
        eps = 1e-7
        f0 = objective(x)
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (objective(xp) - f0) / eps
        return g

    res = minimize(
        objective,
        np.concatenate([bary0.flatten(), [alpha0]]),
        method='SLSQP',
        jac=jac,
        options={'maxiter': 600, 'ftol': 1e-12, 'disp': False}
    )
    if res.success or res.x is not None:
        b = _clamp_bary(res.x[:n].reshape(N, 3))
        a = float(res.x[n])
        # True min area
        true_min = _min_area_bary(b)
        return b, true_min
    return None, -1.0


def _slsqp_refine(bary, deadline_fn):
    """Refine by maximizing alpha starting from current min area."""
    a0 = _min_area_bary(bary)
    b, score = _slsqp_max_alpha(bary, a0, deadline_fn)
    if b is None:
        return bary, a0
    return b, score


def _coordinate_descent_bary(bary, n_iters=400, deadline_fn=None):
    """Coordinate descent in barycentric space: optimize one point at a time."""
    current = bary.copy()
    current_score = _min_area_bary(current)
    step_sizes = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            if deadline_fn is not None and deadline_fn() < 5.0:
                return current, current_score
            improved = False
            for i in range(N):
                base = current[i].copy()
                # perturb b and c (a determined)
                for db in [-step, 0, step]:
                    for dc in [-step, 0, step]:
                        if db == 0 and dc == 0:
                            continue
                        trial = current.copy()
                        trial[i, 1] = base[1] + db
                        trial[i, 2] = base[2] + dc
                        trial = _clamp_bary(trial)
                        ts = _min_area_bary(trial)
                        if ts > current_score + 1e-14:
                            current = trial
                            current_score = ts
                            improved = True
            if not improved:
                break
    return current, current_score


def construct():
    start_time = time.time()
    deadline = 140.0

    def time_left():
        return deadline - (time.time() - start_time)

    best_bary = _hexagonal_init()
    best_score = _min_area_bary(best_bary)

    # Phase 1: Differential evolution (global) with hard objective
    if time_left() > 40.0:
        def de_obj(x):
            b = _clamp_bary(x.reshape(N, 3))
            return -_min_area_bary(b)
        try:
            de_res = differential_evolution(
                de_obj,
                bounds=[(0, 1)] * (N * 3),
                maxiter=60,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='deferred'
            )
            if de_res.x is not None:
                b = _clamp_bary(de_res.x.reshape(N, 3))
                s = _min_area_bary(b)
                if s > best_score:
                    best_bary = b
                    best_score = s
        except Exception:
            pass

    # Phase 2: Multi-start SLSQP alpha-max from diverse inits
    inits = [_hexagonal_init()]
    for seed in [1, 7, 13, 29, 42, 100, 200, 300]:
        inits.append(_random_bary_init(seed))
    # perturb best
    rng = np.random.default_rng(99)
    for k in range(6):
        pert = best_bary + rng.normal(0, 0.015, size=(N, 3))
        inits.append(_clamp_bary(pert))

    for init in inits:
        if time_left() < 12.0:
            break
        b, s = _slsqp_refine(init, time_left)
        if s > best_score:
            best_bary = b
            best_score = s

    # Phase 3: Coordinate descent polish
    if time_left() > 15.0:
        b, s = _coordinate_descent_bary(best_bary, n_iters=600, deadline_fn=time_left)
        if s > best_score:
            best_bary = b
            best_score = s

    # Phase 4: Another SLSQP round after CD
    if time_left() > 12.0:
        b, s = _slsqp_refine(best_bary, time_left)
        if s > best_score:
            best_bary = b
            best_score = s

    # Phase 5: Final coordinate descent
    if time_left() > 8.0:
        b, s = _coordinate_descent_bary(best_bary, n_iters=400, deadline_fn=time_left)
        if s > best_score:
            best_bary = b
            best_score = s

    # Final: ensure feasibility and return cartesian
    best_bary = _clamp_bary(best_bary)
    return _bary_to_cart(best_bary)
