"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines compact structural parameterization (corner + center + rings) with
multi-phase optimization: DE global search on low-dim structural space,
L-BFGS-B polish with smooth soft-max penalty, SLSQP constraint-based polish,
and coordinate-descent radius maximization for final slack extraction.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty.
    Uses quadratic penalty for smoothness + log-sum-exp soft-max for directionality.
    """
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    beta = 60.0

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Log-sum-exp soft-max on the worst violation
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    penalty = 50.0 * quad_penalty + 20.0 * sm
    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Post-processing ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity, conservatively."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _radius_max_descent(pos, radii, deadline, max_iter=200):
    """Coordinate-descent pass to maximize each radius given fixed positions.
    For each circle, compute the maximum radius allowed by boundaries and
    all other circles, then set radius to that maximum (with tiny margin).
    Iterate a few times since increasing one radius reduces others' max.
    """
    radii = radii.copy()
    pos = pos.copy()

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            # Max from boundaries
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            # Max from each other circle
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                allowed = dist - radii[j]
                if allowed < max_r:
                    max_r = allowed
            max_r = max(max_r, 1e-6)
            # Use 99.9% to maintain strict non-overlap
            new_r = max_r * 0.9999
            if new_r > radii[i] * (1.0 + 1e-10):
                radii[i] = new_r
                changed = True
        if not changed:
            break

    return pos, radii


def _polish_lbfgsb(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    base_bounds = []
    for i in range(N):
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        base_bounds.append((1e-4, 0.35))

    best_result = None
    best_score = -np.inf

    rng = np.random.RandomState(42)
    num_tries = 3
    for try_idx in range(num_tries):
        if time.time() > deadline - 5:
            break

        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.01, size=len(x0))
            x_try = x0 + perturbation
            x_try[2*N:] = np.maximum(x_try[2*N:], 1e-4)
            for i in range(N):
                x_try[2*i] = np.clip(x_try[2*i], 0.0, 1.0)
                x_try[2*i+1] = np.clip(x_try[2*i+1], 0.0, 1.0)

        try:
            result = minimize(lambda x: _objective(x, deadline), x_try, method='L-BFGS-B',
                              bounds=base_bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_result = (pos_pp, radii_pp)
        except Exception:
            pass

    if best_result is not None:
        return best_result
    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """Constraint-based SLSQP polish with vectorized constraint functions."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.35)] * N

    def boundary_cons(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        left = p[:, 0] - r
        right = 1.0 - p[:, 0] - r
        bot = p[:, 1] - r
        top = 1.0 - p[:, 1] - r
        return np.concatenate([left, right, bot, top])

    def overlap_cons(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        d = p[:, 0][:, None] - p[:, 0][None, :]
        e = p[:, 1][:, None] - p[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        min_dist = r[:, None] + r[None, :]
        idx = np.triu_indices(N, k=1)
        return (dist[idx] - min_dist[idx]).ravel()

    cons = [
        {'type': 'ineq', 'fun': boundary_cons},
        {'type': 'ineq', 'fun': overlap_cons}
    ]

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    best_result = None
    best_score = -np.inf
    rng = np.random.RandomState(123)

    num_tries = 2
    for try_idx in range(num_tries):
        if time.time() > deadline - 3:
            break

        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.005, size=len(x0))
            x_try = x0 + perturbation
            x_try[2*N:] = np.maximum(x_try[2*N:], 1e-4)
            for i in range(N):
                x_try[2*i] = np.clip(x_try[2*i], 0.0, 1.0)
                x_try[2*i+1] = np.clip(x_try[2*i+1], 0.0, 1.0)

        try:
            result = minimize(neg_sum_rad, x_try, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 300, 'ftol': 1e-10})
            if result.x is not None:
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_result = (pos_pp, radii_pp)
        except Exception:
            pass

    if best_result is not None:
        return best_result
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    struct_bounds = [
        (0.06, 0.28),
        (0.06, 0.32),
        (0.3, 0.7),
        (0.3, 0.7),
        (0.18, 0.45),
        (0.04, 0.16),
        (0.04, 0.16),
        (0.30, 0.55),
        (0.03, 0.12),
        (0.03, 0.12),
    ]

    # ---------- Track A: DE on structural space ----------
    if time.time() < deadline - 80:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=160, popsize=28, seed=7,
                                            tol=1e-10, polish=False,
                                            mutation=(0.35, 1.0), recombination=0.75,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                if time.time() < deadline - 45:
                    pos_p, radii_p = _polish_lbfgsb(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # Second DE run
    if time.time() < deadline - 60:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=100, popsize=22, seed=123,
                                             tol=1e-9, polish=False,
                                             mutation=(0.5, 1.1), recombination=0.85,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
                if time.time() < deadline - 35:
                    pos_p2, radii_p2 = _polish_lbfgsb(pos_s2, radii_s2, deadline)
                    if _is_valid(pos_p2, radii_p2):
                        s2 = np.sum(radii_p2)
                        if s2 > best_score:
                            best_score = s2
                            best_pos = pos_p2.copy()
                            best_radii = radii_p2.copy()
        except Exception:
            pass

    # L-BFGS-B on structural seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.105, 0.43, 0.065, 0.065],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: direct 78-dim from structural seed ----------
    if time.time() < deadline - 35:
        pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
        x0 = np.concatenate([pos_s0.flatten(), radii_s0])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Final high-precision polish ----------
    if best_pos is not None and time.time() < deadline - 15:
        pos_p, radii_p = _polish_lbfgsb(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # ---------- SLSQP cross-check ----------
    if best_pos is not None and time.time() < deadline - 8:
        pos_q, radii_q = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_q, radii_q):
            s = np.sum(radii_q)
            if s > best_score:
                best_score = s
                best_pos = pos_q.copy()
                best_radii = radii_q.copy()

    # ---------- Coordinate-descent radius maximization (final slack squeeze) ----------
    if best_pos is not None and time.time() < deadline - 3:
        pos_cd, radii_cd = _radius_max_descent(best_pos, best_radii, deadline, max_iter=300)
        if _is_valid(pos_cd, radii_cd):
            s = np.sum(radii_cd)
            if s > best_score:
                best_score = s
                best_pos = pos_cd.copy()
                best_radii = radii_cd.copy()

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        side = 5
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
