"""Circle packing in unit square: 26 circles, maximize sum of radii.
Two-phase: (1) vectorized force-directed relaxation with momentum,
(2) constraint-aware local search with radius maximization, SA, and SLSQP polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 130.0  # safety margin before 150s kill


def _vectorized_relax(x, y, r, lr=0.01, lam_ov=10.0, lam_bd=10.0,
                      max_steps=5000, deadline=None, momentum=0.3):
    """Fast vectorized force-directed relaxation with momentum and adaptive LR."""
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    r = r.astype(float).copy()
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        d_safe = np.where(d < 1e-8, 1e-8, d)

        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        np.fill_diagonal(overlap, 0)

        # Overlap forces
        gx_ov = -2 * lam_ov * np.sum(overlap * (dx / d_safe), axis=1)
        gy_ov = -2 * lam_ov * np.sum(overlap * (dy / d_safe), axis=1)
        gr_ov = 2 * lam_ov * np.sum(overlap, axis=1)

        # Boundary forces
        bxl = np.maximum(0, r - x)
        bxr = np.maximum(0, r - (1 - x))
        byl = np.maximum(0, r - y)
        byr = np.maximum(0, r - (1 - y))

        gx_bd = -2 * lam_bd * bxl + 2 * lam_bd * bxr
        gy_bd = -2 * lam_bd * byl + 2 * lam_bd * byr
        gr_bd = 2 * lam_bd * (bxl + bxr + byl + byr)

        # Objective: maximize radii
        gr_obj = -np.ones(N)

        gx = gx_ov + gx_bd
        gy = gy_ov + gy_bd
        gr = gr_ov + gr_bd + gr_obj

        # Adaptive learning rate
        grad_mag = np.sqrt(np.sum(gx**2) + np.sum(gy**2) + np.sum(gr**2))
        if grad_mag > 1e-8:
            eff_lr = lr * min(1.0, 5.0 / grad_mag)
        else:
            eff_lr = lr

        # Update with momentum
        vx = momentum * vx + eff_lr * gx
        vy = momentum * vy + eff_lr * gy
        vr = momentum * vr + eff_lr * gr

        x -= vx
        y -= vy
        r -= vr

        r = np.clip(r, 0.01, 0.4)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        lr *= 0.998

    return x, y, r


def _max_radius_for_position(x, y, r, exclude=None):
    """For each circle i, compute max radius allowed by boundary and
    distance to all other circles. Vectorized."""
    r_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    r_constraint = d - r[None, :]
    r_max = np.minimum(r_max, np.min(r_constraint, axis=1))
    
    r_max = np.maximum(r_max, 1e-6)
    return r_max


def _local_search(x, y, r, deadline, start_time):
    """Constraint-aware local search with multiple phases."""
    x = x.copy()
    y = y.copy()
    r = r.copy()

    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 0: Enforce validity by shrinking
    for _ in range(100):
        if time.time() > deadline:
            break
        r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, r_max_b)
        r = np.maximum(r, 1e-6)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum - 1e-9
        np.fill_diagonal(overlaps, False)

        if not np.any(overlaps):
            break

        overlap_count = np.sum(overlaps, axis=1)
        for i in range(N):
            if overlap_count[i] > 0:
                r[i] *= (1 - 0.03 * overlap_count[i])
        r = np.maximum(r, 1e-6)

    # Phase 1: Iterative radius maximization
    for _ in range(200):
        if time.time() > deadline:
            break
        r_max = _max_radius_for_position(x, y, r)
        r = r_max
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 2: Simulated Annealing on positions with pair-swap moves
    T = 0.015
    T_min = 0.0001
    cooling = 0.975
    sa_steps = 800
    for step in range(sa_steps):
        if time.time() > deadline:
            break
        if T < T_min:
            break

        move_type = np.random.rand()
        old_xi, old_yi = x.copy(), y.copy()

        if move_type < 0.6:
            # Single circle perturbation
            i = np.random.randint(N)
            x[i] += np.random.normal(0, 0.015)
            y[i] += np.random.normal(0, 0.015)
            x[i] = np.clip(x[i], 0.01, 0.99)
            y[i] = np.clip(y[i], 0.01, 0.99)
        elif move_type < 0.85:
            # Pair swap: exchange positions of two circles
            i, j = np.random.choice(N, 2, replace=False)
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]
        else:
            # Two-circle perturbation
            i, j = np.random.choice(N, 2, replace=False)
            for idx in [i, j]:
                x[idx] += np.random.normal(0, 0.01)
                y[idx] += np.random.normal(0, 0.01)
                x[idx] = np.clip(x[idx], 0.01, 0.99)
                y[idx] = np.clip(y[idx], 0.01, 0.99)

        r_new = _max_radius_for_position(x, y, r)
        s_new = np.sum(r_new)
        s_old = np.sum(r)
        delta = s_new - s_old

        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            r = r_new
            if s_new > best_sum:
                best_sum = s_new
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            x, y = old_xi, old_yi

        T *= cooling

    # Phase 3: Pairwise separation moves
    for iteration in range(300):
        if time.time() > deadline:
            break

        r = _max_radius_for_position(x, y, r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        tight = np.abs(d - r_sum) < 0.02
        np.fill_diagonal(tight, False)

        if not np.any(tight):
            break

        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if not tight[i, j]:
                    continue
                dirx = x[i] - x[j]
                diry = y[i] - y[j]
                norm = np.sqrt(dirx**2 + diry**2)
                if norm < 1e-8:
                    continue
                dirx /= norm
                diry /= norm

                delta = 0.004
                new_xi = x[i] + delta * dirx
                new_yi = y[i] + delta * diry
                new_xj = x[j] - delta * dirx
                new_yj = y[j] - delta * diry

                if (new_xi < 0 or new_xi > 1 or new_yi < 0 or new_yi > 1 or
                        new_xj < 0 or new_xj > 1 or new_yj < 0 or new_yj > 1):
                    continue

                old_xi, old_yi = x[i], y[i]
                old_xj, old_yj = x[j], y[j]
                x[i], y[i] = new_xi, new_yi
                x[j], y[j] = new_xj, new_yj

                r_new = _max_radius_for_position(x, y, r)
                s_new = np.sum(r_new)
                if s_new > np.sum(r) + 1e-8:
                    r = r_new
                    moved = True
                else:
                    x[i], y[i] = old_xi, old_yi
                    x[j], y[j] = old_xj, old_yj

        if not moved:
            break

        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 4: Multiple SLSQP optimization passes
    if time.time() < deadline - 10:
        for pass_idx in range(3):
            if time.time() > deadline - 5:
                break
            try:
                def _neg_sum(vars):
                    return -np.sum(vars[2*N:3*N])

                def _constraint_overlap(vars):
                    xx = vars[:N]
                    yy = vars[N:2*N]
                    rr = vars[2*N:3*N]
                    dx = xx[:, None] - xx[None, :]
                    dy = yy[:, None] - yy[None, :]
                    d = np.sqrt(dx**2 + dy**2)
                    r_sum = rr[:, None] + rr[None, :]
                    np.fill_diagonal(d, np.inf)
                    np.fill_diagonal(r_sum, 0)
                    return d - r_sum - 1e-9

                def _constraint_boundary(vars):
                    xx = vars[:N]
                    yy = vars[N:2*N]
                    rr = vars[2*N:3*N]
                    return np.concatenate([
                        xx - rr, 1 - xx - rr, yy - rr, 1 - yy - rr
                    ])

                constraints = [
                    {'type': 'ineq', 'fun': lambda v: np.min(_constraint_overlap(v))},
                    {'type': 'ineq', 'fun': lambda v: np.min(_constraint_boundary(v))},
                ]

                bounds = [(0, 1) for _ in range(N)] + [(0, 1) for _ in range(N)] + [(0.001, 0.4) for _ in range(N)]

                if pass_idx == 0:
                    x0 = np.concatenate([best_x, best_y, best_r])
                elif pass_idx == 1:
                    # Perturbed start
                    x0 = np.concatenate([best_x + np.random.normal(0, 0.005, N),
                                         best_y + np.random.normal(0, 0.005, N),
                                         best_r])
                    x0[:N] = np.clip(x0[:N], 0, 1)
                    x0[N:2*N] = np.clip(x0[N:2*N], 0, 1)
                else:
                    # Current best from previous pass
                    x0 = np.concatenate([best_x, best_y, best_r])

                result = minimize(_neg_sum, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})

                if result.success and _validate(result.x[:N], result.x[N:2*N], result.x[2*N:3*N]):
                    s_opt = np.sum(result.x[2*N:3*N])
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_x, best_y, best_r = (result.x[:N].copy(), result.x[N:2*N].copy(),
                                                   result.x[2*N:3*N].copy())
            except Exception:
                pass

    if _validate(best_x, best_y, best_r):
        return best_x, best_y, best_r
    return x, y, r


def _validate(x, y, r, tol=1e-9):
    """Check validity: boundary and no overlaps."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or \
       np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    if np.any(d < r_sum - tol):
        return False
    return True


def _hex_init(n, scale=1.0):
    """Place n circles on a hexagonal lattice scaled to fit in unit square."""
    s = scale
    pts = []
    cols = int(np.ceil(1.0 / s)) + 2
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    for j in range(rows):
        y = j * s * np.sqrt(3) / 2
        offset = 0 if j % 2 == 0 else s / 2
        for i in range(cols):
            x = i * s + offset
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.rand(n - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    return pts[:n]


def _spiral_init(n):
    """Golden angle spiral initialization."""
    pts = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.35 * np.sqrt(i / n)
        theta = i * golden_angle
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        pts.append((x, y))
    return np.array(pts)


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_score = -np.inf
    best_config = None

    def _try_config(x, y, r):
        nonlocal best_score, best_config
        if _validate(x, y, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_config = (x.copy(), y.copy(), r.copy())
        else:
            for _ in range(50):
                if time.time() > deadline:
                    return
                r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                r = np.minimum(r, r_max_b)
                r = np.maximum(r, 1e-6)
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(d, np.inf)
                r_sum = r[:, None] + r[None, :]
                overlaps = d < r_sum - 1e-9
                np.fill_diagonal(overlaps, False)
                if not np.any(overlaps):
                    break
                overlap_count = np.sum(overlaps, axis=1)
                for i in range(N):
                    if overlap_count[i] > 0:
                        r[i] *= (1 - 0.03 * overlap_count[i])
                r = np.maximum(r, 1e-6)
            if _validate(x, y, r):
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_config = (x.copy(), y.copy(), r.copy())

    # Multiple diverse initializations
    num_restarts = 14
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        np.random.seed(2000 + restart * 173)

        if restart % 5 == 0:
            # Hex lattice
            scale = 0.17 - 0.008 * (restart // 5)
            pts = _hex_init(N, scale)
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            r = np.full(N, 0.03)
        elif restart % 5 == 1:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        px = (i + 0.5) / side + np.random.uniform(-0.06, 0.06)
                        py = (j + 0.5) / side + np.random.uniform(-0.06, 0.06)
                        px = np.clip(px, 0.1, 0.9)
                        py = np.clip(py, 0.1, 0.9)
                        xs.append(px)
                        ys.append(py)
                        rs.append(0.05)
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        elif restart % 5 == 2:
            # Random
            x = np.random.uniform(0.15, 0.85, N)
            y = np.random.uniform(0.15, 0.85, N)
            r = np.random.uniform(0.03, 0.07, N)
        elif restart % 5 == 3:
            # Clustered: few large, many small
            xs, ys, rs = [], [], []
            for _ in range(6):
                xs.append(np.random.uniform(0.2, 0.8))
                ys.append(np.random.uniform(0.2, 0.8))
                rs.append(np.random.uniform(0.07, 0.11))
            for _ in range(N - 6):
                xs.append(np.random.uniform(0.1, 0.9))
                ys.append(np.random.uniform(0.1, 0.9))
                rs.append(np.random.uniform(0.02, 0.05))
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        else:
            # Spiral
            pts = _spiral_init(N)
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            r = np.full(N, 0.04)

        # Phase 1: Force-directed relaxation
        x, y, r = _vectorized_relax(x, y, r, lr=0.01, lam_ov=10.0, lam_bd=10.0,
                                    max_steps=3000, deadline=deadline)
        _try_config(x, y, r)

        if time.time() > deadline:
            break

        # Phase 2: Local search with SA + SLSQP
        x2, y2, r2 = _local_search(x, y, r, deadline, t_start)
        _try_config(x2, y2, r2)

    # Fallback: simple grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_config = (np.array([c[0] for c in cells[:N]]),
                       np.array([c[1] for c in cells[:N]]),
                       np.array([c[2] for c in cells[:N]]))

    x, y, r = best_config
    return np.column_stack((x, y, r))
