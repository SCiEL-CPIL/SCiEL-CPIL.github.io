"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _md_simulate(xy0, r0, deadline, n_steps=20000, dt=0.002, k_rep=500.0,
                 k_wall=800.0, k_pressure=0.5, damping=0.995,
                 r_min=0.005, r_max=0.49):
    """
    Physics-based MD simulation of pressurized bubbles.
    Each circle has position (x,y), radius r, and velocity (vx,vy).
    Forces:
      - Internal pressure: pushes radius outward (dr/dt = k_pressure * r^2 - damping_r)
      - Repulsive contact: soft-sphere potential between overlapping circles
      - Wall: harmonic repulsion from boundaries
    Returns (xy, r, total, valid) for the best configuration seen.
    """
    n = len(xy0)
    # State
    x = xy0[:, 0].copy().astype(np.float64)
    y = xy0[:, 1].copy().astype(np.float64)
    r = r0.copy().astype(np.float64)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vr = np.zeros(n)  # radius velocity

    best_total = 0.0
    best_valid = False
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()

    # Precompute pair list (all pairs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Adaptive dt: start small, can grow
    current_dt = dt
    max_dt = 0.005

    for step in range(n_steps):
        if time.time() > deadline:
            break

        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)

        # Internal pressure: F_r = k_pressure * r^2 (expansion drive)
        # Use a soft form to avoid runaway: F_r = k_pressure * r * (r_max - r)
        for i in range(n):
            fr[i] = k_pressure * r[i] * (r_max - r[i]) * 0.1
            # Wall forces
            # Left wall
            if x[i] < r[i]:
                overlap = r[i] - x[i]
                fx[i] += k_wall * overlap
            # Right wall
            if 1.0 - x[i] < r[i]:
                overlap = r[i] - (1.0 - x[i])
                fx[i] -= k_wall * overlap
            # Bottom wall
            if y[i] < r[i]:
                overlap = r[i] - y[i]
                fy[i] += k_wall * overlap
            # Top wall
            if 1.0 - y[i] < r[i]:
                overlap = r[i] - (1.0 - y[i])
                fy[i] -= k_wall * overlap

        # Pairwise repulsive forces (soft-sphere)
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap = min_dist - dist
                # Force magnitude: k_rep * overlap^2 (soft)
                f_mag = k_rep * overlap * overlap
                inv_dist = 1.0 / dist
                fx[i] += f_mag * dx * inv_dist
                fy[i] += f_mag * dy * inv_dist
                fx[j] -= f_mag * dx * inv_dist
                fy[j] -= f_mag * dy * inv_dist
                # Also push radii apart slightly
                fr[i] -= 0.5 * k_rep * overlap * 0.01
                fr[j] -= 0.5 * k_rep * overlap * 0.01

        # Velocity Verlet integration
        # Update velocities
        vx = vx * damping + fx * current_dt
        vy = vy * damping + fy * current_dt
        vr = vr * damping + fr * current_dt

        # Limit velocities
        max_v = 0.05
        vx = np.clip(vx, -max_v, max_v)
        vy = np.clip(vy, -max_v, max_v)
        vr = np.clip(vr, -0.01, 0.01)

        # Update positions
        x = x + vx * current_dt
        y = y + vy * current_dt
        r = r + vr * current_dt

        # Clamp radii
        r = np.clip(r, r_min, r_max)

        # Hard-clamp positions to be inside square (with radius margin)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])

        # Check validity and track best
        if step % 50 == 0:
            total, valid = _eval_packing(np.column_stack([x, y]), r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()

        # Adaptive dt: if energy is high, reduce dt
        kinetic = np.sum(vx * vx + vy * vy)
        if kinetic > 0.01:
            current_dt = max(0.0005, current_dt * 0.9)
        elif kinetic < 0.001:
            current_dt = min(max_dt, current_dt * 1.05)

        # Cool down: reduce pressure over time to allow settling
        if step > n_steps * 0.5:
            k_pressure *= 0.9995

    # Final evaluation
    xy_final = np.column_stack([best_x, best_y])
    total, valid = _eval_packing(xy_final, best_r)
    if valid:
        return xy_final, best_r, total, True
    return best_x, best_y, best_r, best_valid


def _md_with_annealing(xy0, r0, deadline):
    """
    Run MD simulation with multiple restarts at different temperature/pressure levels.
    Returns best (xy, r, total, valid).
    """
    best_xy = xy0.copy()
    best_r = r0.copy()
    best_total = 0.0
    best_valid = False

    # Try different pressure strengths and step counts
    configs = [
        {'k_pressure': 0.3, 'n_steps': 8000, 'dt': 0.003},
        {'k_pressure': 0.5, 'n_steps': 10000, 'dt': 0.002},
        {'k_pressure': 0.8, 'n_steps': 12000, 'dt': 0.002},
        {'k_pressure': 1.2, 'n_steps': 15000, 'dt': 0.0015},
    ]

    for cfg in configs:
        if time.time() > deadline:
            break
        xy, r, total, valid = _md_simulate(
            xy0, r0, deadline,
            n_steps=cfg['n_steps'], dt=cfg['dt'],
            k_pressure=cfg['k_pressure']
        )
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n):
    """Smooth objective with softplus penalties."""
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    beta = 30.0
    wall_pen = np.sum(np.log1p(np.exp(beta * wall_viol)) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.log1p(np.exp(beta * viol)) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n):
    """Gradient of smooth objective."""
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()
    grad = np.zeros(3 * n)
    beta = 30.0

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sigmoid_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        sw = sigmoid_w[i]
        if sw > 1e-10:
            dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = int(np.argmin(dists))
            grad[2*n + i] += 100.0 * sw
            if min_idx == 0:
                grad[i] -= 100.0 * sw
            elif min_idx == 1:
                grad[i] += 100.0 * sw
            elif min_idx == 2:
                grad[n + i] -= 100.0 * sw
            else:
                grad[n + i] += 100.0 * sw

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s > 1e-10:
                factor = 100.0 * s
                if dist > 1e-12:
                    grad[i] += factor * dx / dist
                    grad[n + i] += factor * dy / dist
                    grad[j] -= factor * dx / dist
                    grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0
    return grad


def _gradient_polish(xy, r, deadline):
    """L-BFGS-B polish on smooth objective."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.499))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxfun': 20000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
    return xy, r


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _hex_init():
    """Hexagonal packing initialization."""
    positions = []
    # Hexagonal grid
    rows = 5
    cols = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48

    for row in range(rows):
        for col in range(cols):
            if len(positions) >= N:
                break
            x = dx * (col + 0.5) + (row % 2) * dx * 0.5
            y = dy * (row + 0.5)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y, r_base])

    while len(positions) < N:
        positions.append([0.5, 0.5, 0.02])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple initializations + MD simulation
    inits = []

    # Grid inits
    for side in [5, 6, 7]:
        arr = _grid_init(side)
        inits.append((arr[:, :2], arr[:, 2].copy()))

    # Hex init
    hex_arr = _hex_init()
    inits.append((hex_arr[:, :2], hex_arr[:, 2].copy()))

    # Random inits
    for seed in range(6):
        xy, r = _random_init(N, seed=seed)
        inits.append((xy, r))

    for init_xy, init_r in inits:
        if time.time() > deadline:
            break
        out_xy, out_r, total, valid = _md_with_annealing(init_xy, init_r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy.copy()
            best_r = out_r.copy()

    # Phase 2: Gradient polish on best result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 3: Another MD round from polished state
        if time.time() < deadline - 5:
            md_xy, md_r, md_total, md_valid = _md_with_annealing(
                best_xy, best_r, deadline)
            if md_valid and md_total > best_total:
                best_total = md_total
                best_xy = md_xy.copy()
                best_r = md_r.copy()

            # Final gradient polish
            if time.time() < deadline - 2:
                polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
                total, valid = _eval_packing(polished_xy, polished_r)
                if valid and total > best_total:
                    best_total = total
                    best_xy = polished_xy
                    best_r = polished_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        arr = _grid_init(side)
        xy = arr[:, :2]
        r = arr[:, 2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
