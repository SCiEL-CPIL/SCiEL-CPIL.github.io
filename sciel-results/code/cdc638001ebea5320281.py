"""Circle packing in unit square - 26 circles maximizing sum of radii.
Optimized: vectorized post-processing, cached objectives, better time allocation.
"""
import numpy as np
import time
from scipy.optimize import minimize

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


def _objective(x, deadline=None):
    """Objective: minimize negative sum of radii + smooth penalty."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

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
    beta = 60.0
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    # Combined penalty
    penalty = 50.0 * quad_penalty + 20.0 * sm

    return -np.sum(radii) + penalty


def _post_process(pos, radii, deadline=None, max_iters=100):
    """Iteratively shrink circles to ensure validity (vectorized)."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        valid = True

        # Boundary check (vectorized)
        max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over_bound = radii > max_r
        if np.any(over_bound):
            radii[over_bound] = max_r[over_bound] * 0.999
            valid = False

        # Overlap check (vectorized)
        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        min_dist = radii[:, None] + radii[None, :]
        viol = min_dist - dist
        np.fill_diagonal(viol, 0.0)

        # Find pairs with violation
        for i in range(N):
            for j in range(i + 1, N):
                if viol[i, j] > 1e-9:
                    if radii[i] <= radii[j]:
                        needed = viol[i, j] / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.7, 1.0 - needed * 0.99)
                    else:
                        needed = viol[i, j] / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.7, 1.0 - needed * 0.99)
                    valid = False

        if valid:
            break

    # Final boundary clamp
    max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                      np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.maximum(1e-6, np.minimum(radii, max_r * 0.999))

    return pos, radii


def _lbfgs_polish(pos, radii, time_budget=15.0):
    """L-BFGS-B polishing with tighter tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))

    start = time.time()
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _slsqp_polish(pos, radii, time_budget=10.0):
    """SLSQP polishing with hard constraints."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))

    # Define constraints for SLSQP
    def _boundary_constraints(x):
        n = N
        pos_c = x[:2*n].reshape(n, 2)
        radii_c = x[2*n:]
        # Each circle must be inside [0,1]^2
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})  # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})  # x_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})  # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})  # y_i + r_i <= 1
        return cons

    def _overlap_constraints(x):
        n = N
        pos_c = x[:2*n].reshape(n, 2)
        radii_c = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def _dist_constraint(x, i=i, j=j):
                    pi = x[2*i:2*i+2]
                    pj = x[2*j:2*j+2]
                    ri = x[2*n+i]
                    rj = x[2*n+j]
                    return np.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2) - ri - rj
                cons.append({'type': 'ineq', 'fun': _dist_constraint})
        return cons

    constraints = _boundary_constraints(x0) + _overlap_constraints(x0)

    start = time.time()
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-14})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _force_directed(pos, radii, time_budget=8.0, seed=42):
    """Force-directed relaxation: push overlapping circles apart, then grow radii."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    start = time.time()
    step = 0.003
    n_steps = 5000

    for it in range(n_steps):
        if time.time() - start > time_budget:
            break

        # Compute forces from overlaps
        forces = np.zeros((N, 2))
        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        np.fill_diagonal(dist, 1.0)
        min_dist = radii[:, None] + radii[None, :]
        viol = np.maximum(min_dist - dist, 0.0)
        np.fill_diagonal(viol, 0.0)

        # Force magnitude proportional to violation
        for i in range(N):
            for j in range(i + 1, N):
                if viol[i, j] > 0:
                    direction = np.array([d[i, j], e[i, j]]) / max(dist[i, j], 1e-10)
                    f = step * viol[i, j] * 0.5
                    forces[i] += f * direction
                    forces[j] -= f * direction

        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += step * (radii[i] - pos[i, 0]) * 2
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= step * (pos[i, 0] - (1.0 - radii[i])) * 2
            if pos[i, 1] < radii[i]:
                forces[i, 1] += step * (radii[i] - pos[i, 1]) * 2
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= step * (pos[i, 1] - (1.0 - radii[i])) * 2

        # Apply forces
        pos += forces
        pos[:, 0] = np.clip(pos[:, 0], 0.005, 0.995)
        pos[:, 1] = np.clip(pos[:, 1], 0.005, 0.995)

        # Gradually grow radii
        grow_rate = 0.0001 * (1.0 + 0.5 * np.sin(it * 0.01))
        max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        radii = np.minimum(radii + grow_rate, max_r * 0.999)
        radii = np.maximum(radii, 0.005)

        # Post-process every 50 steps
        if it % 50 == 0:
            pos, radii = _post_process(pos, radii)

        if _is_valid(pos, radii):
            s = float(np.sum(radii))
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()

    return best_pos, best_radii, best_score


def _nelder_mead_radius_polish(pos, radii, time_budget=8.0):
    """Nelder-Mead optimization on radii only (positions fixed)."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_score = float(np.sum(radii))
    best_radii = radii.copy()

    start = time.time()
    # Simple coordinate descent + small perturbations
    for it in range(2000):
        if time.time() - start > time_budget:
            break

        improved = False
        for i in range(N):
            # Try increasing radius
            for delta in [0.002, 0.001, 0.0005]:
                new_r = radii.copy()
                new_r[i] += delta
                max_r = np.minimum(np.minimum(pos[i, 0], 1.0 - pos[i, 0]),
                                  np.minimum(pos[i, 1], 1.0 - pos[i, 1]))
                if new_r[i] > max_r * 0.999:
                    continue
                if _is_valid(pos, new_r):
                    s = float(np.sum(new_r))
                    if s > best_score:
                        best_score = s
                        best_radii = new_r.copy()
                        radii = new_r.copy()
                        improved = True
                        break

        if not improved and it % 100 == 0:
            # Try small random perturbations
            rng = np.random.RandomState(it)
            for i in range(N):
                new_r = radii.copy()
                new_r[i] += rng.uniform(-0.001, 0.002)
                max_r = np.minimum(np.minimum(pos[i, 0], 1.0 - pos[i, 0]),
                                  np.minimum(pos[i, 1], 1.0 - pos[i, 1]))
                new_r[i] = np.clip(new_r[i], 0.005, max_r * 0.999)
                if _is_valid(pos, new_r):
                    s = float(np.sum(new_r))
                    if s > best_score:
                        best_score = s
                        best_radii = new_r.copy()
                        radii = new_r.copy()

    return pos, best_radii, best_score


def _anneal(pos, radii, time_budget=20.0, seed=42):
    """Simulated annealing with force-directed phases and reduced post-processing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    T = 0.01
    T_min = 1e-7
    cooling = 0.997
    max_iter = 30000

    start = time.time()
    post_process_every = 5

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Every 500 iterations, do a short force-directed phase
        if it % 500 == 0 and it > 0:
            fd_budget = min(2.0, time_budget - (time.time() - start))
            if fd_budget > 0.5:
                pos_fd, radii_fd, score_fd = _force_directed(pos, radii, time_budget=fd_budget, seed=seed + it)
                if score_fd > best_score:
                    best_score = score_fd
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
                pos, radii = pos_fd, radii_fd
                continue

        move_type = rng.random()
        if move_type < 0.45:
            i = rng.randint(N)
            dx = rng.normal(0, 0.015)
            dy = rng.normal(0, 0.015)
            pos[i, 0] = np.clip(pos[i, 0] + dx, 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1] + dy, 0.005, 0.995)
        elif move_type < 0.75:
            i = rng.randint(N)
            dr = rng.normal(0, 0.004)
            radii[i] = np.clip(radii[i] + dr, 0.005, 0.35)
        elif move_type < 0.90:
            i, j = rng.choice(N, size=2, replace=False)
            pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, 0.015), 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, 0.015), 0.005, 0.995)
            pos[j, 0] = np.clip(pos[j, 0] + rng.normal(0, 0.015), 0.005, 0.995)
            pos[j, 1] = np.clip(pos[j, 1] + rng.normal(0, 0.015), 0.005, 0.995)
        else:
            i, j = rng.choice(N, size=2, replace=False)
            pos[i], pos[j] = pos[j].copy(), pos[i].copy()
            radii[i], radii[j] = radii[j], radii[i]

        if it % post_process_every == 0:
            pos, radii = _post_process(pos, radii)

        x = np.concatenate([pos.flatten(), radii])
        new_obj = _objective(x)
        new_score = -new_obj

        if new_score > best_score - 0.005:
            if _is_valid(pos, radii):
                s = float(np.sum(radii))
                if s > best_score:
                    best_score = s
                    best_pos = pos.copy()
                    best_radii = radii.copy()

        delta = new_score - best_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            pass
        else:
            if rng.random() < 0.3:
                pass

        T *= cooling
        if T < T_min:
            T = T_min

    # Final post-process
    pos, radii = _post_process(pos, radii)
    if _is_valid(pos, radii):
        s = float(np.sum(radii))
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_radii = radii.copy()

    return best_pos, best_radii, best_score


def _make_grid(seed=42):
    """5x5 grid with one extra circle."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.003
    pos[25] = [0.5, 0.02]
    radii[25] = 0.015
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3)
        if y > 0.9:
            break
        offset = r if row % 2 == 0 else 2 * r
        for col in range(5):
            x = offset + col * 2 * r
            if x > 0.9:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05
        idx += 1
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.08, size=N)
    return pos, radii


def _make_center_ring(seed=42):
    """Center + inner ring + outer ring."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
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


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_pos = None
    best_radii = None
    best_score = -np.inf

    # Focus on fewer, higher-quality strategies
    strategies = [
        ('grid', 42),
        ('hexagonal', 42),
        ('center_ring', 42),
        ('random', 42),
        ('grid', 123),
        ('hexagonal', 99),
        ('random', 7),
    ]

    for strat_idx, (strat_name, seed) in enumerate(strategies):
        remaining = deadline - time.time()
        if remaining < 25:
            break

        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(seed)
        else:
            continue

        # Allocate more time to earlier strategies
        frac = 1.0 - 0.1 * strat_idx
        anneal_budget = min(15.0 * frac, remaining * 0.45)
        pos_a, radii_a, score_a = _anneal(pos, radii, time_budget=anneal_budget, seed=seed)

        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()

        # L-BFGS-B polish
        remaining = deadline - time.time()
        if remaining < 12:
            continue
        polish_budget = min(10.0 * frac, remaining * 0.35)
        pos_p, radii_p, score_p = _lbfgs_polish(pos_a, radii_a, time_budget=polish_budget)

        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()

        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 10:
            continue
        slsqp_budget = min(6.0 * frac, remaining * 0.25)
        pos_s, radii_s, score_s = _slsqp_polish(pos_p, radii_p, time_budget=slsqp_budget)

        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()

    # Final high-precision polish on best result
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 25:
            pos_f, radii_f, score_f = _lbfgs_polish(best_pos, best_radii, time_budget=min(15.0, remaining - 10))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()

        remaining = deadline - time.time()
        if remaining > 15:
            pos_s2, radii_s2, score_s2 = _slsqp_polish(best_pos, best_radii, time_budget=min(8.0, remaining - 8))
            if score_s2 > best_score:
                best_score = score_s2
                best_pos = pos_s2.copy()
                best_radii = radii_s2.copy()

        # Nelder-Mead radius-only polish to squeeze out remaining slack
        remaining = deadline - time.time()
        if remaining > 10:
            pos_nm, radii_nm, score_nm = _nelder_mead_radius_polish(best_pos, best_radii, time_budget=min(7.0, remaining - 5))
            if score_nm > best_score:
                best_score = score_nm
                best_pos = pos_nm.copy()
                best_radii = radii_nm.copy()

        # One more L-BFGS pass on the radius-polished result
        remaining = deadline - time.time()
        if remaining > 8:
            pos_f2, radii_f2, score_f2 = _lbfgs_polish(best_pos, best_radii, time_budget=min(5.0, remaining - 3))
            if score_f2 > best_score:
                best_score = score_f2
                best_pos = pos_f2.copy()
                best_radii = radii_f2.copy()

    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
