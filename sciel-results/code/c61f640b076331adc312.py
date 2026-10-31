"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses trust-constr NLP with explicit nonlinear constraints, plus multi-start
Differential Evolution in a compact structural space, and a force-directed
annealing polish to escape local minima."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint

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
    """Objective: minimize negative sum of radii + smooth penalty. Vectorized."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations (vectorized) - smooth quadratic
    left = np.maximum(0.0, radii - pos[:, 0])
    right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bot = np.maximum(0.0, radii - pos[:, 1])
    top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left * left) + np.sum(right * right) + np.sum(bot * bot) + np.sum(top * top)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = np.maximum(0.0, min_dist - dist)
    np.fill_diagonal(viol, 0.0)
    open_pen = 0.5 * np.sum(viol * viol)

    return -np.sum(radii) + 100.0 * (bpen + open_pen)


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

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


# ---------------- NLP constraints for trust-constr ----------------

def _nlp_objective(x, deadline):
    """Minimize -sum(radii) for trust-constr."""
    if time.time() > deadline:
        return 1e10
    radii = x[2 * N:]
    return -np.sum(radii)


def _nlp_bounds_obj(x):
    """Boundary constraints: for each circle i,
    x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0"""
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    # 4 constraints per circle
    c = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    return c


def _nlp_overlap_obj(x):
    """Overlap constraints: for each pair (i,j), dist(i,j) - r_i - r_j >= 0"""
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    c = dist[iu] - min_dist[iu]
    return c


def _nlp_bounds_jac(x):
    """Jacobian of boundary constraints."""
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    # Shape: (4n, 4n)
    jac = np.zeros((4 * n, 4 * n))
    for i in range(n):
        # x_i - r_i >= 0
        jac[4 * i, 2 * i] = 1.0
        jac[4 * i, 2 * n + i] = -1.0
        # 1 - x_i - r_i >= 0
        jac[4 * i + 1, 2 * i] = -1.0
        jac[4 * i + 1, 2 * n + i] = -1.0
        # y_i - r_i >= 0
        jac[4 * i + 2, 2 * i + 1] = 1.0
        jac[4 * i + 2, 2 * n + i] = -1.0
        # 1 - y_i - r_i >= 0
        jac[4 * i + 3, 2 * i + 1] = -1.0
        jac[4 * i + 3, 2 * n + i] = -1.0
    return jac


def _nlp_overlap_jac(x):
    """Jacobian of overlap constraints (upper triangle pairs)."""
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    n_pairs = n * (n - 1) // 2
    jac = np.zeros((n_pairs, 4 * n))
    iu = np.triu_indices(n, k=1)
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-12:
            dist = 1e-12
        # d(dist)/dx_i = dx/dist, d(dist)/dy_i = dy/dist
        # d(dist)/dx_j = -dx/dist, d(dist)/dy_j = -dy/dist
        # d(dist)/dr_i = -1, d(dist)/dr_j = -1
        jac[k, 2 * i] = dx / dist
        jac[k, 2 * i + 1] = dy / dist
        jac[k, 2 * j] = -dx / dist
        jac[k, 2 * j + 1] = -dy / dist
        jac[k, 2 * n + i] = -1.0
        jac[k, 2 * n + j] = -1.0
    return jac


def _solve_nlp(x0, deadline, maxiter=500):
    """Solve NLP with trust-constr and explicit nonlinear constraints."""
    try:
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((1e-4, 0.35))

        # Nonlinear constraints
        cons_boundary = NonlinearConstraint(_nlp_bounds_obj, 0.0, np.inf,
                                             jac=_nlp_bounds_jac)
        cons_overlap = NonlinearConstraint(_nlp_overlap_obj, 0.0, np.inf,
                                            jac=_nlp_overlap_jac)

        result = minimize(_nlp_objective, x0, method='trust-constr',
                          bounds=bounds, constraints=[cons_boundary, cons_overlap],
                          args=(deadline,),
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10})
        return result.x
    except Exception:
        return None


# ---------------- Structural seeds ----------------

def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
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


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _force_directed_anneal(pos, radii, deadline, max_iter=3000, rng_seed=123):
    """Force-directed annealing with vectorized forces and adaptive radius growth."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(rng_seed)

    T = 0.04
    cooling = 0.9995
    min_T = 1e-7

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < min_T:
            break

        forces = np.zeros_like(pos)

        bpen_grad_x = np.where(pos[:, 0] < radii, (pos[:, 0] - radii), 0.0) + \
                       np.where(pos[:, 0] > 1.0 - radii, (pos[:, 0] - (1.0 - radii)), 0.0)
        bpen_grad_y = np.where(pos[:, 1] < radii, (pos[:, 1] - radii), 0.0) + \
                       np.where(pos[:, 1] > 1.0 - radii, (pos[:, 1] - (1.0 - radii)), 0.0)
        forces[:, 0] -= bpen_grad_x * 20.0
        forces[:, 1] -= bpen_grad_y * 20.0

        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        np.fill_diagonal(dist, 1e10)
        min_dist = radii[:, None] + radii[None, :]

        overlap = np.maximum(0.0, min_dist - dist)
        near = np.where((dist > min_dist) & (dist < min_dist * 1.3),
                        (min_dist * 1.3 - dist) * 0.15, 0.0)

        f_mag = overlap * 80.0 / dist + near / dist
        np.fill_diagonal(f_mag, 0.0)

        forces[:, 0] += np.sum(f_mag * d, axis=1)
        forces[:, 1] += np.sum(f_mag * e, axis=1)

        step_size = 0.008 * (T / 0.04)
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                delta = step_size * (forces[i] / f_norm + rng.randn(2) * 0.8)
            else:
                delta = step_size * rng.randn(2) * 0.8
            pos[i] += delta
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)

        if it % 10 == 0:
            for i in range(N):
                boundary_clear = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                                     pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
                min_circle_clear = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_ij = np.sqrt(dx * dx + dy * dy)
                    clear_ij = dist_ij - radii[i] - radii[j]
                    if clear_ij < min_circle_clear:
                        min_circle_clear = clear_ij
                total_clear = min(boundary_clear, min_circle_clear)
                if total_clear > 0.002:
                    growth = min(0.005, total_clear * 0.3)
                    new_r = radii[i] + growth
                    max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                    radii[i] = min(new_r, max_r * 0.999)

        if it % 5 == 0:
            pos_pp, radii_pp = _post_process(pos, radii, deadline)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

        T *= cooling

    pos_pp, radii_pp = _post_process(pos, radii, deadline)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()

    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

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

    # ---------- Track A: NLP with trust-constr from structural seeds ----------
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.10, 0.10, 0.43, 0.065, 0.065],
    ]

    for s0 in struct_seeds:
        if time.time() > deadline - 40:
            break
        try:
            pos_s, radii_s = _expand_struct(np.array(s0))
            x0 = np.concatenate([pos_s.flatten(), radii_s])
            x_nlp = _solve_nlp(x0, deadline, maxiter=300)
            if x_nlp is not None:
                pos_nlp = x_nlp[:2 * N].reshape(N, 2)
                radii_nlp = x_nlp[2 * N:]
                update_best(pos_nlp, radii_nlp)
        except Exception:
            pass

    # ---------- Track B: DE over structural space ----------
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

    if time.time() < deadline - 50:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=100, popsize=18, seed=7,
                                            tol=1e-8, polish=False,
                                            mutation=(0.4, 1.0), recombination=0.7,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                # Polish with NLP
                if time.time() < deadline - 30:
                    x0 = np.concatenate([pos_s.flatten(), radii_s])
                    x_nlp = _solve_nlp(x0, deadline, maxiter=200)
                    if x_nlp is not None:
                        pos_nlp = x_nlp[:2 * N].reshape(N, 2)
                        radii_nlp = x_nlp[2 * N:]
                        update_best(pos_nlp, radii_nlp)
        except Exception:
            pass

    if time.time() < deadline - 35:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                              args=(deadline,),
                                              maxiter=80, popsize=15, seed=42,
                                              tol=1e-8, polish=False,
                                              mutation=(0.6, 1.0), recombination=0.8,
                                              updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
                if time.time() < deadline - 20:
                    x0 = np.concatenate([pos_s2.flatten(), radii_s2])
                    x_nlp = _solve_nlp(x0, deadline, maxiter=200)
                    if x_nlp is not None:
                        pos_nlp = x_nlp[:2 * N].reshape(N, 2)
                        radii_nlp = x_nlp[2 * N:]
                        update_best(pos_nlp, radii_nlp)
        except Exception:
            pass

    # ---------- Track C: Direct 78-dim from seeds ----------
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
    configs.append((pos_s0, radii_s0))

    for pos, radii in configs:
        if time.time() > deadline - 25:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Track D: Force-directed annealing ----------
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline,
                                                       max_iter=3000, rng_seed=456)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_fd2, radii_fd2 = _force_directed_anneal(best_pos, best_radii, deadline,
                                                         max_iter=2000, rng_seed=789)
            if _is_valid(pos_fd2, radii_fd2):
                s2 = np.sum(radii_fd2)
                if s2 > best_score:
                    best_score = s2
                    best_pos = pos_fd2.copy()
                    best_radii = radii_fd2.copy()
        except Exception:
            pass

    # ---------- Final NLP polish ----------
    if best_pos is not None and time.time() < deadline - 5:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            x_nlp = _solve_nlp(x0, deadline, maxiter=200)
            if x_nlp is not None:
                pos_nlp = x_nlp[:2 * N].reshape(N, 2)
                radii_nlp = x_nlp[2 * N:]
                update_best(pos_nlp, radii_nlp)
        except Exception:
            pass

    # ---------- Final L-BFGS-B polish ----------
    if best_pos is not None and time.time() < deadline - 3:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
