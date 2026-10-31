"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses structural parameterization, force-directed annealing, and multi-phase polish.
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


def _expand_struct(s):
    """Expand structural parameters (center positions + radius ratios) to full config.
    s has shape (2*N,): first 2*N are positions, then N radius ratios.
    We use a base radius and distribute area proportionally."""
    n = N
    pos = s[:2 * n].reshape(n, 2)
    # Use positive radii via softplus-like transform on last N entries
    raw_r = s[2 * n:]
    radii = np.exp(raw_r - 3.0)  # softplus approx: exp(x-3) gives ~0.05 at x=0
    radii = np.clip(radii, 0.005, 0.4)
    return pos, radii


def _struct_objective(s, deadline=None):
    """Objective in structural space."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    pos, radii = _expand_struct(s)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


def _post_process(pos, radii, deadline=None, max_iters=200):
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

        # Find pairs with violation - shrink the smaller circle less aggressively
        for i in range(N):
            for j in range(i + 1, N):
                v = viol[i, j]
                if v > 1e-9:
                    # Shrink both proportionally to their radii
                    total_r = radii[i] + radii[j]
                    if total_r < 1e-12:
                        continue
                    frac_i = radii[i] / total_r
                    frac_j = radii[j] / total_r
                    # Need to reduce sum of radii by v, split proportionally
                    reduce_i = v * frac_i * 1.05  # 5% extra margin
                    reduce_j = v * frac_j * 1.05
                    radii[i] = max(0.001, radii[i] - reduce_i)
                    radii[j] = max(0.001, radii[j] - reduce_j)
                    valid = False

        if valid:
            break

    # Final boundary clamp
    max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                      np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.maximum(1e-6, np.minimum(radii, max_r * 0.999))

    return pos, radii


def _gradient_polish(pos, radii, time_budget=5.0):
    """Fast gradient-based polish: move circles to reduce violations."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    start = time.time()
    lr = 0.02
    n_iter = 500

    for it in range(n_iter):
        if time.time() - start > time_budget:
            break

        # Compute forces from overlaps and boundaries
        forces = np.zeros((N, 2))

        # Overlap forces
        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e + 1e-12)
        min_dist = radii[:, None] + radii[None, :]
        viol = min_dist - dist
        np.fill_diagonal(viol, 0.0)

        for i in range(N):
            for j in range(i + 1, N):
                if viol[i, j] > 0:
                    # Push apart
                    if dist[i, j] < 1e-10:
                        fx, fy = 0.01, 0.01
                    else:
                        fx = d[i, j] / dist[i, j] * viol[i, j]
                        fy = e[i, j] / dist[i, j] * viol[i, j]
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 2
            if pos[i, 0] > 1 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1 - radii[i])) * 2
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 2
            if pos[i, 1] > 1 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1 - radii[i])) * 2

        # Update positions
        pos += lr * forces
        pos = np.clip(pos, 0.005, 0.995)

        # Try to grow radii slightly if no violations
        pos, radii = _post_process(pos, radii)
        # Attempt small growth
        if it % 20 == 0:
            max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                              np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
            target_r = np.minimum(max_r * 0.99, radii + 0.001)
            radii = np.clip(target_r, 0.001, 0.4)
            pos, radii = _post_process(pos, radii)

        if _is_valid(pos, radii):
            s = float(np.sum(radii))
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()

        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.8

    return best_pos, best_radii, best_score


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
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 30000})
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
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
        return cons

    def _overlap_constraints(x):
        n = N
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
                          options={'maxiter': 300, 'ftol': 1e-14})
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


def _force_directed_anneal(pos, radii, time_budget=20.0, seed=42):
    """Force-directed simulated annealing: physics-based moves + SA acceptance."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    T = 0.005
    T_min = 1e-8
    cooling = 0.998
    max_iter = 50000

    start = time.time()

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Compute current forces
        forces = np.zeros((N, 2))
        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e + 1e-12)
        min_dist = radii[:, None] + radii[None, :]
        viol = min_dist - dist
        np.fill_diagonal(viol, 0.0)

        for i in range(N):
            for j in range(i + 1, N):
                if viol[i, j] > 0:
                    if dist[i, j] < 1e-10:
                        fx, fy = 0.01, 0.01
                    else:
                        fx = d[i, j] / dist[i, j] * viol[i, j]
                        fy = e[i, j] / dist[i, j] * viol[i, j]
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 3
            if pos[i, 0] > 1 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1 - radii[i])) * 3
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 3
            if pos[i, 1] > 1 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1 - radii[i])) * 3

        # Force-directed move
        step = 0.01 * (1.0 + T * 100)
        pos += step * forces
        pos = np.clip(pos, 0.005, 0.995)

        # Random perturbation scaled by temperature
        if rng.random() < 0.5:
            i = rng.randint(N)
            pos[i] += rng.normal(0, 0.01 * (1 + T * 50), size=2)
            pos[i] = np.clip(pos[i], 0.005, 0.995)

        # Occasionally try radius growth
        if it % 50 == 0:
            i = rng.randint(N)
            max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            if radii[i] < max_r * 0.95:
                radii[i] = min(radii[i] * 1.005, max_r * 0.99)

        pos, radii = _post_process(pos, radii)

        if _is_valid(pos, radii):
            s = float(np.sum(radii))
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()
                # Reheat slightly on improvement
                T = max(T, 0.002)

        # SA acceptance: always accept improvements, accept worsenings with prob exp(-delta/T)
        # Since we track best separately, we just keep going
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


def _make_staggered(seed=42):
    """Staggered grid: alternating rows offset by half cell."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    rows = 5
    cols = 6
    r = 1.0 / (2 * cols) - 0.002
    idx = 0
    for i in range(rows):
        y = (i + 0.5) / rows
        offset = 0.5 / cols if i % 2 == 0 else 0.0
        for j in range(cols):
            x = offset + (j + 0.5) / cols
            if x > 0.99:
                continue
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.03
        idx += 1
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
    deadline = start_time + 138.0  # Safety margin before 150s kill

    best_pos = None
    best_radii = None
    best_score = -np.inf

    # Diverse initial configurations
    strategies = [
        ('grid', 42),
        ('hexagonal', 42),
        ('center_ring', 42),
        ('staggered', 42),
        ('random', 42),
        ('grid', 123),
        ('hexagonal', 99),
        ('staggered', 7),
    ]

    for strat_idx, (strat_name, seed) in enumerate(strategies):
        remaining = deadline - time.time()
        if remaining < 30:
            break

        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(seed)
        elif strat_name == 'staggered':
            pos, radii = _make_staggered(seed)
        else:
            continue

        # Allocate time: more for earlier strategies
        frac = 1.0 - 0.08 * strat_idx
        anneal_budget = min(18.0 * frac, remaining * 0.40)
        pos_a, radii_a, score_a = _force_directed_anneal(pos, radii, time_budget=anneal_budget, seed=seed)

        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()

        # Gradient polish (fast)
        remaining = deadline - time.time()
        if remaining < 15:
            continue
        grad_budget = min(6.0 * frac, remaining * 0.20)
        pos_g, radii_g, score_g = _gradient_polish(pos_a, radii_a, time_budget=grad_budget)

        if score_g > best_score:
            best_score = score_g
            best_pos = pos_g.copy()
            best_radii = radii_g.copy()

        # L-BFGS-B polish
        remaining = deadline - time.time()
        if remaining < 12:
            continue
        polish_budget = min(12.0 * frac, remaining * 0.30)
        pos_p, radii_p, score_p = _lbfgs_polish(pos_g, radii_g, time_budget=polish_budget)

        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()

        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 10:
            continue
        slsqp_budget = min(8.0 * frac, remaining * 0.25)
        pos_s, radii_s, score_s = _slsqp_polish(pos_p, radii_p, time_budget=slsqp_budget)

        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()

    # Final high-precision polish on best result
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 30:
            pos_f, radii_f, score_f = _lbfgs_polish(best_pos, best_radii, time_budget=min(20.0, remaining - 12))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()

        remaining = deadline - time.time()
        if remaining > 18:
            pos_s2, radii_s2, score_s2 = _slsqp_polish(best_pos, best_radii, time_budget=min(12.0, remaining - 6))
            if score_s2 > best_score:
                best_score = score_s2
                best_pos = pos_s2.copy()
                best_radii = radii_s2.copy()

        # Final gradient polish squeeze
        remaining = deadline - time.time()
        if remaining > 8:
            pos_g2, radii_g2, score_g2 = _gradient_polish(best_pos, best_radii, time_budget=min(6.0, remaining - 3))
            if score_g2 > best_score:
                best_score = score_g2
                best_pos = pos_g2.copy()
                best_radii = radii_g2.copy()

    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
