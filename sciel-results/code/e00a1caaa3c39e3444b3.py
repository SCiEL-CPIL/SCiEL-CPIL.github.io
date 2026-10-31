"""Circle packing in unit square - 26 circles maximizing sum of radii.
Structural change: free-position DE + fast radial gradient growth,
replacing the fixed ring parameterization. Positions are explored by
DE in 39 dims; radii are grown by an iterative greedy+gradient loop."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
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
    """Penalized objective: -sum(radii) + penalty for violations."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations
    left = np.maximum(0.0, radii - pos[:, 0])
    right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bot = np.maximum(0.0, radii - pos[:, 1])
    top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))

    # Overlap violations
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = np.maximum(0.0, min_dist - dist)
    np.fill_diagonal(viol, 0.0)

    all_v = np.concatenate([left, right, bot, top, viol.ravel()])
    quad = np.sum(all_v ** 2)

    beta = 60.0
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    penalty = 50.0 * quad + 20.0 * sm
    return -np.sum(radii) + penalty


def _grow_radii(pos, radii, deadline, max_iters=500, lr=0.02):
    """Iterative radial gradient ascent: grow each radius toward its
    maximum allowed by boundary and pairwise constraints."""
    radii = radii.copy()
    pos = pos.copy()
    n = N

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Compute max allowed radius for each circle
        max_r = np.zeros(n)
        for i in range(n):
            max_r[i] = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])

        # Pairwise constraints: r_i + r_j <= dist_ij
        # For each i, the tightest constraint from pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dist = 1e-12
                # r_i <= dist - r_j
                limit_i = dist - radii[j]
                limit_j = dist - radii[i]
                if limit_i < max_r[i]:
                    max_r[i] = limit_i
                if limit_j < max_r[j]:
                    max_r[j] = limit_j

        # Grow radii toward max_r
        improved = False
        for i in range(n):
            target = max(0.001, max_r[i])
            if radii[i] < target:
                new_r = radii[i] + lr * (target - radii[i])
                if new_r > radii[i] + 1e-12:
                    radii[i] = new_r
                    improved = True
            elif radii[i] > target:
                radii[i] = max(0.001, target * 0.999)
                improved = True

        if not improved:
            break

    # Final clamp
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return radii


def _grow_radii_with_position(pos, radii, deadline, max_iters=300, lr=0.015):
    """Grow radii while also nudging positions to relieve tight constraints."""
    radii = radii.copy()
    pos = pos.copy()
    n = N

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Compute max allowed radius for each circle
        max_r = np.zeros(n)
        for i in range(n):
            max_r[i] = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])

        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dist = 1e-12
                limit_i = dist - radii[j]
                limit_j = dist - radii[i]
                if limit_i < max_r[i]:
                    max_r[i] = limit_i
                if limit_j < max_r[j]:
                    max_r[j] = limit_j

        # Grow radii
        improved = False
        for i in range(n):
            target = max(0.001, max_r[i])
            if radii[i] < target:
                new_r = radii[i] + lr * (target - radii[i])
                if new_r > radii[i] + 1e-12:
                    radii[i] = new_r
                    improved = True
            elif radii[i] > target:
                radii[i] = max(0.001, target * 0.999)
                improved = True

        # Nudge positions: push apart circles that are tight
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist * 1.1 and dist > 1e-12:
                    # Push apart
                    push = 0.002 * (min_dist * 1.1 - dist) / dist
                    pos[i, 0] += push * dx
                    pos[i, 1] += push * dy
                    pos[j, 0] -= push * dx
                    pos[j, 1] -= push * dy
                    pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
                    pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
                    pos[j, 0] = np.clip(pos[j, 0], 0.01, 0.99)
                    pos[j, 1] = np.clip(pos[j, 1], 0.01, 0.99)

        if not improved:
            break

    # Final clamp
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


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
    """L-BFGS-B polish on full 78-dim problem."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.005, 0.35))
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _make_grid(n):
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


def _make_hexagonal(n):
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


def _make_center_ring(n):
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


def _struct_objective(p, deadline):
    """Objective for structural parameterization (kept as fallback)."""
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
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r
    pos[4] = [cx, cy]
    radii[4] = center_r
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


def _free_pos_objective(p, deadline):
    """DE objective: p = 26 positions (52 dims). Radii are grown greedily."""
    if time.time() > deadline:
        return 1e10
    pos = p.reshape(N, 2)
    # Initial small radii
    radii = np.full(N, 0.02)
    # Grow radii
    radii = _grow_radii(pos, radii, deadline, max_iters=200, lr=0.03)
    # Check validity
    if not _is_valid(pos, radii):
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if not _is_valid(pos_pp, radii_pp):
            return 1e6
        return -np.sum(radii_pp)
    return -np.sum(radii)


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

    # ---------- Phase 1: Seed configurations + radial growth ----------
    seeds = [
        _make_grid(N),
        _make_hexagonal(N),
        _make_center_ring(N),
        _make_random(N, seed=1),
        _make_random(N, seed=7),
        _make_random(N, seed=42),
    ]

    for pos, radii in seeds:
        if time.time() > deadline - 80:
            break
        # Grow radii from seed
        radii_grown = _grow_radii(pos, radii, deadline, max_iters=300, lr=0.02)
        update_best(pos, radii_grown)

    # ---------- Phase 2: DE on free positions (52-dim) ----------
    if time.time() < deadline - 50:
        try:
            pos_bounds = []
            for _ in range(N):
                pos_bounds.append((0.05, 0.95))
                pos_bounds.append((0.05, 0.95))

            de_res = differential_evolution(_free_pos_objective, pos_bounds,
                                            args=(deadline,),
                                            maxiter=80, popsize=25, seed=7,
                                            tol=1e-6, polish=False,
                                            mutation=(0.5, 1.0), recombination=0.8,
                                            updating='deferred')
            if de_res.x is not None:
                pos_de = de_res.x.reshape(N, 2)
                radii_de = _grow_radii(pos_de, np.full(N, 0.02), deadline, max_iters=300, lr=0.02)
                update_best(pos_de, radii_de)
        except Exception:
            pass

    # ---------- Phase 3: Second DE with different seed/params ----------
    if time.time() < deadline - 35:
        try:
            pos_bounds = []
            for _ in range(N):
                pos_bounds.append((0.05, 0.95))
                pos_bounds.append((0.05, 0.95))

            de_res2 = differential_evolution(_free_pos_objective, pos_bounds,
                                             args=(deadline,),
                                             maxiter=60, popsize=20, seed=123,
                                             tol=1e-6, polish=False,
                                             mutation=(0.7, 1.3), recombination=0.7,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_de2 = de_res2.x.reshape(N, 2)
                radii_de2 = _grow_radii(pos_de2, np.full(N, 0.02), deadline, max_iters=300, lr=0.02)
                update_best(pos_de2, radii_de2)
        except Exception:
            pass

    # ---------- Phase 4: Structural DE (10-dim, kept from parent) ----------
    if time.time() < deadline - 30:
        struct_bounds = [
            (0.06, 0.28), (0.06, 0.32), (0.3, 0.7), (0.3, 0.7),
            (0.18, 0.45), (0.04, 0.16), (0.04, 0.16),
            (0.30, 0.55), (0.03, 0.12), (0.03, 0.12),
        ]
        try:
            de_res3 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=100, popsize=20, seed=99,
                                             tol=1e-7, polish=False,
                                             mutation=(0.4, 1.0), recombination=0.7,
                                             updating='deferred')
            if de_res3.x is not None:
                # Expand structural
                corner_r = np.clip(de_res3.x[0], 0.05, 0.30)
                center_r = np.clip(de_res3.x[1], 0.05, 0.35)
                cx = np.clip(de_res3.x[2], 0.2, 0.8)
                cy = np.clip(de_res3.x[3], 0.2, 0.8)
                r1_off = np.clip(de_res3.x[4], 0.15, 0.6)
                r1_r = np.clip(de_res3.x[5], 0.03, 0.20)
                r1_r2 = np.clip(de_res3.x[6], 0.03, 0.20)
                r2_off = np.clip(de_res3.x[7], 0.25, 0.7)
                r2_r = np.clip(de_res3.x[8], 0.02, 0.16)
                r2_r2 = np.clip(de_res3.x[9], 0.02, 0.16)
                pos_s = np.zeros((N, 2))
                radii_s = np.zeros(N)
                pos_s[0] = [corner_r, corner_r]
                pos_s[1] = [1.0 - corner_r, corner_r]
                pos_s[2] = [corner_r, 1.0 - corner_r]
                pos_s[3] = [1.0 - corner_r, 1.0 - corner_r]
                radii_s[0:4] = corner_r
                pos_s[4] = [cx, cy]
                radii_s[4] = center_r
                for i in range(8):
                    ang = 2 * np.pi * i / 8.0
                    pos_s[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
                    radii_s[5 + i] = r1_r if i % 2 == 0 else r1_r2
                for i in range(13):
                    ang = 2 * np.pi * i / 13.0 + 0.1
                    pos_s[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
                    radii_s[13 + i] = r2_r if i % 2 == 0 else r2_r2
                # Grow radii from structural positions
                radii_grown = _grow_radii(pos_s, radii_s, deadline, max_iters=200, lr=0.02)
                update_best(pos_s, radii_grown)
        except Exception:
            pass

    # ---------- Phase 5: L-BFGS-B polish on best ----------
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_p, radii_p = _polish(best_pos, best_radii, deadline)
            if _is_valid(pos_p, radii_p):
                s = np.sum(radii_p)
                if s > best_score:
                    best_score = s
                    best_pos = pos_p.copy()
                    best_radii = radii_p.copy()
        except Exception:
            pass

    # ---------- Phase 6: Grow radii with position nudging ----------
    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_np, radii_np = _grow_radii_with_position(best_pos, best_radii, deadline,
                                                          max_iters=200, lr=0.01)
            if _is_valid(pos_np, radii_np):
                s = np.sum(radii_np)
                if s > best_score:
                    best_score = s
                    best_pos = pos_np.copy()
                    best_radii = radii_np.copy()
        except Exception:
            pass

    # ---------- Phase 7: Final L-BFGS-B polish ----------
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_f, radii_f = _polish(best_pos, best_radii, deadline)
            if _is_valid(pos_f, radii_f):
                s = np.sum(radii_f)
                if s > best_score:
                    best_score = s
                    best_pos = pos_f.copy()
                    best_radii = radii_f.copy()
        except Exception:
            pass

    # ---------- Phase 8: Final radial growth ----------
    if best_pos is not None and time.time() < deadline - 2:
        try:
            radii_final = _grow_radii(best_pos, best_radii, deadline, max_iters=100, lr=0.01)
            if _is_valid(best_pos, radii_final):
                s = np.sum(radii_final)
                if s > best_score:
                    best_score = s
                    best_radii = radii_final.copy()
        except Exception:
            pass

    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
