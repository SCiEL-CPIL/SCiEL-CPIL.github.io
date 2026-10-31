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

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)

    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Smooth boundary violations
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)

    boundary_pen = np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) + \
                   np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta))

    # Smooth overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_logsumexp(x, penalty_weight=100.0, beta=50.0):
    """
    Objective using log-sum-exp for smooth max(0, v) approximation.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)

    # Overlap violations using log-sum-exp for smooth max
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol_raw = min_dist_sq - dist_sq
    viol_masked = viol_raw[mask]

    # Log-sum-exp: log(sum(exp(beta * v))) / beta approximates max(v)
    if len(viol_masked) > 0:
        max_v = np.max(viol_masked)
        if max_v > 0:
            overlap_pen = (np.log(np.sum(np.exp(beta * (viol_masked - max_v)))) + beta * max_v) / beta
        else:
            overlap_pen = 0.0
    else:
        overlap_pen = 0.0

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_polish(pos, radii, deadline, maxiter=500):
    """
    SLSQP polish with explicit hard constraints for non-overlap and boundary.
    """
    x = np.concatenate([pos.flatten(), radii])
    n = len(radii)

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.9999))  # x
        bounds.append((1e-4, 0.9999))  # y
    for i in range(n):
        bounds.append((1e-4, 0.49))  # r

    # Inequality constraints: g(x) >= 0
    def _boundary_constraints(x):
        pos_c = x[:2 * n].reshape(n, 2)
        r_c = x[2 * n:]
        cons = []
        for i in range(n):
            cons.append(pos_c[i, 0] - r_c[i])
            cons.append((1.0 - pos_c[i, 0]) - r_c[i])
            cons.append(pos_c[i, 1] - r_c[i])
            cons.append((1.0 - pos_c[i, 1]) - r_c[i])
        return np.array(cons)

    def _overlap_constraints(x):
        pos_c = x[:2 * n].reshape(n, 2)
        r_c = x[2 * n:]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist_sq = dx * dx + dy * dy
                sum_r = r_c[i] + r_c[j]
                cons.append(dist_sq - sum_r * sum_r)
        return np.array(cons)

    constraints = [
        {'type': 'ineq', 'fun': _boundary_constraints},
        {'type': 'ineq', 'fun': _overlap_constraints}
    ]

    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success or result.fun < _objective(x):
            x = result.x
    except Exception:
        pass

    pos_out = x[:2 * n].reshape(n, 2)
    radii_out = np.maximum(x[2 * n:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    return pos_out, radii_out

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(200):
        valid = True

        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False

        if valid:
            break

    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)

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
            best_x, best_y = 0.5, 0.5
            best_r = 0.003

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97

    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

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

def _initial_corner_biased(seed=7):
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

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 50.0)

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac

        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)

        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr

        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)

        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r

        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 50.0)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r

        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

    return best_pos, best_radii

def _try_radius_expansion(pos, radii, max_iterations=200):
    """Try to expand each circle's radius while maintaining validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)

            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])

            new_r = max_r_circ * 0.999999
            if new_r > r * 1.00001:
                radii[i] = new_r
                if _is_valid(pos, radii, tol=1e-8):
                    improved = True
                else:
                    radii[i] = r

        if not improved:
            break

    return pos, radii

def _perturbation_local_search(pos, radii, n_perturbations=500, max_step=0.02, seed=0, deadline=None):
    """Local search with random perturbations."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 2:
            break

        n_pert = rng.randint(1, 4)
        indices = rng.choice(n, size=n_pert, replace=False)

        old_pos = pos[indices].copy()
        old_radii = radii[indices].copy()

        for idx in indices:
            dx = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dy = rng.normal(0, max_step * rng.uniform(0.5, 1.0))
            dr = rng.normal(0, max_step * 0.3)

            new_x = pos[idx, 0] + dx
            new_y = pos[idx, 1] + dy
            new_r = radii[idx] + dr

            new_x = np.clip(new_x, 0.005, 0.995)
            new_y = np.clip(new_y, 0.005, 0.995)
            new_r = np.clip(new_r, 0.005, 0.4)

            pos[idx, 0] = new_x
            pos[idx, 1] = new_y
            radii[idx] = new_r

        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos.copy()
                radii = best_radii.copy()
        else:
            pos[indices] = old_pos
            radii[indices] = old_radii

    return best_pos, best_radii

def _force_directed_relax(pos, radii, n_steps=500, dt=0.001, deadline=None):
    """
    Force-directed relaxation: treat circles as particles with repulsion forces.
    """
    rng = np.random.RandomState(42)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 2:
            break

        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                sum_r = radii[i] + radii[j]
                if dist < sum_r * 1.5:
                    f = 10.0 * (sum_r - dist) / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy

            x, y = pos[i]
            r = radii[i]
            if x < r:
                forces[i, 0] += 5.0 * (r - x)
            if x > 1 - r:
                forces[i, 0] -= 5.0 * (x - (1 - r))
            if y < r:
                forces[i, 1] += 5.0 * (r - y)
            if y > 1 - r:
                forces[i, 1] -= 5.0 * (y - (1 - r))

        pos += dt * forces
        pos = np.clip(pos, 0.005, 0.995)

        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    return best_pos, best_radii

def _jitter_and_expand(pos, radii, deadline, n_rounds=200):
    """
    Jitter-and-expand: randomly perturb positions slightly, then solve a QP-like
    radius maximization (via SLSQP with fixed positions) to inflate radii.
    Accept if the expanded configuration has higher sum of radii.
    """
    rng = np.random.RandomState(12345)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for rnd in range(n_rounds):
        if time.time() > deadline - 3:
            break

        # Jitter: perturb 2-4 circle positions by a small amount
        n_jit = rng.randint(2, 5)
        indices = rng.choice(n, size=n_jit, replace=False)
        old_pos = pos[indices].copy()

        for idx in indices:
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = rng.uniform(0.002, 0.015)
            pos[idx, 0] += magnitude * np.cos(angle)
            pos[idx, 1] += magnitude * np.sin(angle)
            pos[idx, 0] = np.clip(pos[idx, 0], 0.01, 0.99)
            pos[idx, 1] = np.clip(pos[idx, 1], 0.01, 0.99)

        # Expand: solve for optimal radii given fixed positions
        # Use SLSQP to maximize sum(r) subject to non-overlap and boundary
        r_init = radii.copy()
        x_r = r_init.copy()

        bounds_r = [(0.005, 0.49)] * n

        def _rad_objective(x_r):
            r = np.maximum(x_r, 1e-6)
            # Penalty for violations
            pen = 0.0
            for i in range(n):
                pen += max(0, r[i] - pos[i, 0]) ** 2
                pen += max(0, pos[i, 0] - (1 - r[i])) ** 2
                pen += max(0, r[i] - pos[i, 1]) ** 2
                pen += max(0, pos[i, 1] - (1 - r[i])) ** 2
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    s = r[i] + r[j]
                    if d < s:
                        pen += (s - d) ** 2
            return -np.sum(r) + 50.0 * pen

        def _rad_boundary(x_r):
            r = x_r
            cons = []
            for i in range(n):
                cons.append(pos[i, 0] - r[i])
                cons.append((1 - pos[i, 0]) - r[i])
                cons.append(pos[i, 1] - r[i])
                cons.append((1 - pos[i, 1]) - r[i])
            return np.array(cons)

        def _rad_overlap(x_r):
            r = x_r
            cons = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d2 = dx * dx + dy * dy
                    s = r[i] + r[j]
                    cons.append(d2 - s * s)
            return np.array(cons)

        try:
            result = minimize(_rad_objective, x_r, method='SLSQP',
                              bounds=bounds_r,
                              constraints=[
                                  {'type': 'ineq', 'fun': _rad_boundary},
                                  {'type': 'ineq', 'fun': _rad_overlap}
                              ],
                              options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                new_radii = np.maximum(result.x, 1e-5)
            else:
                new_radii = r_init
        except Exception:
            new_radii = r_init

        # Validate
        pos_pp, radii_pp = _post_process(pos, new_radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos.copy()
                radii = best_radii.copy()
            else:
                # Revert positions
                pos[indices] = old_pos
        else:
            pos[indices] = old_pos

    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multi-stage local search."""
    x = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))

    # Stage 1: SLSQP with explicit hard constraints
    if time.time() < deadline - 35:
        pos_out, radii_out = _slsqp_polish(pos, radii, deadline, maxiter=500)
        pos = pos_out
        radii = radii_out
        x = np.concatenate([pos.flatten(), radii])

    # Stage 2: L-BFGS-B with high penalty on smooth objective
    if time.time() < deadline - 25:
        try:
            result = minimize(_objective_smooth, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 40.0),
                              options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-11})
            if result.fun < _objective_smooth(x, 5000.0, 40.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    # Stage 3: Log-sum-exp objective polish with higher beta
    if time.time() < deadline - 18:
        try:
            result = minimize(_objective_logsumexp, x, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0, 80.0),
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-11})
            if result.fun < _objective_logsumexp(x, 10000.0, 80.0) - 1e-12:
                x = result.x
        except Exception:
            pass

    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-5)
    pos_out, radii_out = _post_process(pos_out, radii_out)

    # Stage 4: Second SLSQP pass with tighter constraints
    if time.time() < deadline - 12:
        pos_out, radii_out = _slsqp_polish(pos_out, radii_out, deadline, maxiter=400)

    # Stage 5: Force-directed relaxation to escape local minima
    if time.time() < deadline - 8:
        pos_out, radii_out = _force_directed_relax(pos_out, radii_out, n_steps=300, deadline=deadline)

    # Stage 6: Jitter-and-expand
    if time.time() < deadline - 6:
        pos_out, radii_out = _jitter_and_expand(pos_out, radii_out, deadline, n_rounds=150)

    # Stage 7: Aggressive radius expansion
    if time.time() < deadline - 4:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=300)

    # Stage 8: Perturbation local search
    if time.time() < deadline - 3:
        pos_out, radii_out = _perturbation_local_search(
            pos_out, radii_out,
            n_perturbations=200, max_step=0.008, seed=42,
            deadline=deadline
        )

    # Stage 9: Final tight SLSQP
    if time.time() < deadline - 2:
        pos_out, radii_out = _slsqp_polish(pos_out, radii_out, deadline, maxiter=300)

    # Stage 10: Final radius expansion
    if time.time() < deadline - 1:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)

    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))

    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))

    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))

    pos_cb2, rad_cb2 = _initial_corner_biased(seed=13)
    inits.append((pos_cb2, rad_cb2))

    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
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

    penalty_stages = [50.0, 500.0, 2000.0]

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break

        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        # Stage 1: Standard objective
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
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

        # Stage 2: Smooth objective with log-sum-exp
        if time.time() < deadline - 10:
            for p_w, beta in [(500.0, 30.0), (2000.0, 40.0)]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, beta),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, beta) - 1e-10:
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

            # Log-sum-exp variant
            if time.time() < deadline - 5:
                for p_w, beta in [(1000.0, 50.0), (3000.0, 70.0)]:
                    if time.time() > deadline - 3:
                        break
                    try:
                        result = minimize(_objective_logsumexp, current_x, method='L-BFGS-B', bounds=bounds,
                                          args=(p_w, beta),
                                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                        if result.fun < _objective_logsumexp(current_x, p_w, beta) - 1e-10:
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

        # Stage 3: Simulated annealing
        if time.time() < deadline - 8:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    n_steps=2000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
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

    # Extra SLSQP polish if time permits
    if time.time() < deadline - 3:
        try:
            pos_extra, rad_extra = _slsqp_polish(best_pos, best_radii, deadline, maxiter=300)
            pos_pp, radii_pp = _post_process(pos_extra, rad_extra)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final radius expansion squeeze
    if time.time() < deadline - 1:
        try:
            pos_final, radii_final = _try_radius_expansion(best_pos, best_radii, max_iterations=200)
            pos_pp, radii_pp = _post_process(pos_final, radii_final)
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
