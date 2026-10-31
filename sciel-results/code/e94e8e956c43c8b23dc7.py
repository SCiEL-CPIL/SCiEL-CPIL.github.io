"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=1000.0, use_smooth=False, beta=50.0):
    """
    Objective function: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # 1. Boundary violations
    viol_left = np.maximum(0.0, radii - pos[:, 0])
    viol_right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    viol_bottom = np.maximum(0.0, radii - pos[:, 1])
    viol_top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))

    if use_smooth:
        boundary_pen = (np.sum(_soft_max(viol_left, beta)) + np.sum(_soft_max(viol_right, beta)) +
                        np.sum(_soft_max(viol_bottom, beta)) + np.sum(_soft_max(viol_top, beta)))
    else:
        boundary_pen = np.sum(viol_left ** 2) + np.sum(viol_right ** 2) + \
                       np.sum(viol_bottom ** 2) + np.sum(viol_top ** 2)

    # 2. Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist

    # Only consider upper triangle to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    # Violation is when min_dist_sq > dist_sq
    viol_overlap = min_dist_sq - dist_sq

    if use_smooth:
        overlap_pen = np.sum(_soft_max(viol_overlap[mask], beta))
    else:
        viol_overlap_clipped = np.maximum(0.0, viol_overlap)
        overlap_pen = np.sum(viol_overlap_clipped[mask] ** 2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # dist_sq >= min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii and adjust positions to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(100):
        valid = True

        # Fix boundaries
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False

        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii down to fit
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False

        if valid:
            break

    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # Start with a reasonable radius
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5

    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r

        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx

            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1

        if k >= N:
            break

    # Fill remaining with random small circles
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1

    return pos, radii

def _init_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18

    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)

        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])

        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2

        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95

    return pos, radii

def _init_best_fit(seed=42):
    """Greedy best-fit initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # First circle in center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if not found or best_r < 0.003:
            # Fallback: place small circle
            best_x, best_y = 0.5, 0.5
            best_r = 0.003

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97

    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=2000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated annealing.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    x = np.concatenate([pos.flatten(), radii])
    current_energy = _objective(x, 500.0, use_smooth=False)

    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac

        # Perturb one circle
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)

        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr

        # Clamp
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)

        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r

        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 500.0, use_smooth=False)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r

        # Check validity and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))

    # SLSQP
    try:
        result = minimize(lambda z: _objective(z, 5000.0, use_smooth=False), x, method='SLSQP', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 5000.0, use_smooth=False) - 1e-12:
            x = result.x
    except Exception:
        pass

    # L-BFGS-B
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda z: _objective(z, 10000.0, use_smooth=False), x, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 10000.0, use_smooth=False) - 1e-12:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-4)

    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = []
    pos_bf, rad_bf = _init_hexagonal()
    inits.append((pos_bf, rad_bf))

    pos_cb, rad_cb = _init_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))

    pos_fit, rad_fit = _init_best_fit(seed=42)
    inits.append((pos_fit, rad_fit))

    for i in range(2):
        pos_r, rad_r = _init_random(100 + i)
        inits.append((pos_r, rad_r))

    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)

    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))

    # Stages: (penalty_weight, use_smooth, beta)
    penalty_stages = [
        (100.0, False, 50.0),
        (500.0, False, 50.0),
        (2000.0, False, 50.0),
        (5000.0, True, 30.0),  # Smooth pass for final refinement
    ]

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break

        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        # Gradient based optimization
        for p_w, smooth, beta in penalty_stages:
            if time.time() > deadline - 15:
                break
            try:
                if smooth:
                    func = lambda z: _objective(z, p_w, use_smooth=True, beta=beta)
                else:
                    func = lambda z: _objective(z, p_w, use_smooth=False)

                result = minimize(func, current_x, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < func(current_x) - 1e-10:
                    current_x = result.x
            except Exception:
                pass

            pos_opt = current_x[:2 * N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2 * N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

        # Simulated Annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    n_steps=1500, t0=0.03, t_end=0.0005,
                    seed=idx * 100, deadline=deadline - 10
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final polish
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)

    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
