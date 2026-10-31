"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to
maximize the smallest triangle area. Barycentric optimization with soft-min and
epigraph surrogates plus a high-precision local polish."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 143.0  # Safety margin for 150s wall-clock limit

# Pre-compute triangle indices (165 triangles)
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def calc_areas_bary(bary):
    """Areas of all triangles formed by N points, in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]

    det = (ui * (vj * wk - vk * wj)
           - vi * (uj * wk - uk * wj)
           + wi * (uj * vk - uk * vj))
    return np.abs(det)


def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    return np.column_stack([u, v])


def hard_min_obj(bary_flat):
    """Objective using hard min (nonsmooth)."""
    bary = clamp_bary(bary_flat.reshape(N, 2))
    return -np.min(calc_areas_bary(bary))


def soft_min_obj(bary_flat, beta=50.0):
    """Smooth surrogate for the hard min via log-sum-exp."""
    bary = clamp_bary(bary_flat.reshape(N, 2))
    areas = calc_areas_bary(bary)

    if np.any(areas < 1e-15):
        return 1e10

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -(-log_sum / beta)


def epigraph_obj(x, beta=200.0):
    """Epigraph form: x = [bary_flat (22), t]. Minimize -t with a soft penalty
    pushing every triangle area up to t. Equivalent to max-min but smooth."""
    bary = clamp_bary(x[:N * 2].reshape(N, 2))
    t = x[N * 2]
    areas = calc_areas_bary(bary)
    deficit = np.maximum(0.0, t - areas)
    return -t + beta * np.sum(deficit ** 2)


def generate_hexagonal_init():
    """Hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf

    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        for k in range(int(1.0 / scale) + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])

        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()

    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary


def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])


def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)


def _time_left(deadline, reserve=0.0):
    return deadline - time.time() - reserve


def refine_local(bary_init, deadline):
    """Multi-stage refinement: soft-min ladder -> epigraph -> hard-min NM polish."""
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()

    def better(new_bary):
        nonlocal bary_current
        s = -hard_min_obj(new_bary.flatten())
        c = -hard_min_obj(bary_current.flatten())
        if s > c:
            bary_current = clamp_bary(new_bary)
        return bary_current

    # Stage 1-3: graduated soft-min
    for beta, maxiter in [(20.0, 200), (50.0, 200), (200.0, 300)]:
        if _time_left(deadline, 2.0) <= 0:
            break
        try:
            r = minimize(soft_min_obj, bary_current.flatten(), args=(beta,),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-9})
            if r.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = better(r.x.reshape(N, 2))
        except Exception:
            pass

    # Stage 4: epigraph refinement (smooth, well-conditioned)
    for beta, maxiter in [(200.0, 300), (800.0, 300)]:
        if _time_left(deadline, 2.0) <= 0:
            break
        t0 = max(1e-6, np.min(calc_areas_bary(bary_current)))
        x0 = np.concatenate([bary_current.flatten(), [t0]])
        ep_bounds = [(0, 1)] * (N * 2) + [(-1e-6, 1.0)]
        try:
            r = minimize(epigraph_obj, x0, args=(beta,),
                         method='L-BFGS-B', bounds=ep_bounds,
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            cand = r.x[:N * 2].reshape(N, 2)
            if -hard_min_obj(cand.flatten()) > -hard_min_obj(bary_current.flatten()):
                bary_current = better(cand)
        except Exception:
            pass

    # Stage 5-6: hard-min Nelder-Mead polish (adaptive)
    for maxiter, xatol, fatol in [(1000, 1e-10, 1e-12), (800, 1e-12, 1e-14)]:
        if _time_left(deadline, 2.0) <= 0:
            break
        try:
            r = minimize(hard_min_obj, bary_current.flatten(), method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': xatol,
                                  'fatol': fatol, 'adaptive': True})
            if r.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = better(r.x.reshape(N, 2))
        except Exception:
            pass

    return bary_current


def refine_epigraph(bary_init, deadline):
    """Dedicated epigraph refinement with a high-precision final polish."""
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()

    # Warm up with soft-min
    if _time_left(deadline, 2.0) > 0:
        try:
            r = minimize(soft_min_obj, bary_current.flatten(), args=(200.0,),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if r.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(r.x.reshape(N, 2))
        except Exception:
            pass

    # Epigraph ladder
    for beta, maxiter in [(200.0, 300), (800.0, 400), (3000.0, 400)]:
        if _time_left(deadline, 2.0) <= 0:
            break
        t0 = max(1e-6, np.min(calc_areas_bary(bary_current)))
        x0 = np.concatenate([bary_current.flatten(), [t0]])
        ep_bounds = [(0, 1)] * (N * 2) + [(-1e-6, 1.0)]
        try:
            r = minimize(epigraph_obj, x0, args=(beta,),
                         method='L-BFGS-B', bounds=ep_bounds,
                         options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11})
            cand = r.x[:N * 2].reshape(N, 2)
            if -hard_min_obj(cand.flatten()) > -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(cand)
        except Exception:
            pass

    # High-precision hard-min polish
    for maxiter, xatol, fatol in [(1500, 1e-11, 1e-13), (1000, 1e-13, 1e-15)]:
        if _time_left(deadline, 2.0) <= 0:
            break
        try:
            r = minimize(hard_min_obj, bary_current.flatten(), method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': xatol,
                                  'fatol': fatol, 'adaptive': True})
            if r.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(r.x.reshape(N, 2))
        except Exception:
            pass

    return bary_current


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_score = -float('inf')

    # ---- Build diverse initial configurations ----
    initial_configs = []
    hex_init = generate_hexagonal_init()
    initial_configs.append(hex_init)

    for seed in [1, 2, 3, 4, 5, 6, 7]:
        initial_configs.append(generate_random_init(seed))

    for sigma in [0.02, 0.04, 0.07, 0.10]:
        initial_configs.append(generate_perturbed_init(hex_init, sigma,
                                                       seed=100 + int(sigma * 100)))

    # Symmetric-ish inits: centroid + ring
    cx, cy = 0.5, 1.0 / 3.0
    for r_frac in [0.25, 0.35, 0.45]:
        ring = []
        for k in range(10):
            ang = 2 * np.pi * k / 10
            # place on a circle in barycentric space around centroid
            u = cx + r_frac * np.cos(ang) * 0.5
            v = cy + r_frac * np.sin(ang) * 0.5
            ring.append([u, v])
        ring.append([cx, cy])
        bary = clamp_bary(np.array(ring))
        initial_configs.append(bary)

    # ---- Refine each config, tracking the best ----
    for i, init_bary in enumerate(initial_configs):
        if _time_left(deadline, 8.0) <= 0:
            break
        remaining = _time_left(deadline, 8.0)
        # spend a bounded chunk per config, but allow more as configs finish
        per = max(1.0, min(remaining / max(1, len(initial_configs) - i), 12.0))
        cfg_deadline = time.time() + per
        try:
            refined = refine_local(init_bary, cfg_deadline)
        except Exception:
            refined = init_bary
        score = -hard_min_obj(refined.flatten())
        if score > best_score:
            best_score = score
            best_bary = refined.copy()

    # ---- Global search via differential evolution if time remains ----
    if _time_left(deadline, 15.0) > 0:
        try:
            bounds = [(0, 1)] * (N * 2)
            de_deadline = deadline - 5.0

            def de_wrapper(x):
                if time.time() > de_deadline:
                    return 1e10
                return hard_min_obj(x)

            de_result = differential_evolution(
                de_wrapper,
                bounds=bounds,
                maxiter=25,
                popsize=18,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol',
                update_interval=1,
            )
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            de_score = -hard_min_obj(de_bary.flatten())
            if de_score > best_score:
                best_score = de_score
                best_bary = de_bary.copy()
        except Exception:
            pass

    # ---- Final high-precision polish on the best solution ----
    if _time_left(deadline, 4.0) > 0 and best_bary is not None:
        try:
            final = refine_epigraph(best_bary, deadline)
            final_score = -hard_min_obj(final.flatten())
            if final_score > best_score:
                best_score = final_score
                best_bary = final.copy()
        except Exception:
            pass

    # ---- Fallback ----
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = clamp_bary(np.column_stack([u, v]))

    return bary_to_cart(best_bary)
