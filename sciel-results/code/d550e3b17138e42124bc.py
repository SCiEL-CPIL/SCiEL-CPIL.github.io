"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# Deadline guard: 145s wall-clock (5s safety margin under 150s)
_DEADLINE = 145.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum.reduce([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]])
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _is_valid(x, n=N, tol=1e-6):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _grow_radii(x, n=N, max_iter=300, step=0.0008, tol=1e-8):
    """
    Try to grow radii while maintaining feasibility (greedy per-circle increase).
    """
    pts = x.reshape(n, 3).copy()
    for _ in range(max_iter):
        if _remaining() < 2.0:
            break
        improved = False
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        for i in range(n):
            max_r = min(xs[i], ys[i], 1.0 - xs[i], 1.0 - ys[i])
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                max_r = min(max_r, d - rs[j])
            new_r = min(max_r, rs[i] + step)
            if new_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = new_r
                pts[i, 2] = new_r
                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r
        if not improved:
            break
    return pts.ravel()


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est
    xs, ys = [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        x_start = (1.0 - total_width) / 2.0
        for j in range(count):
            xs.append(x_start + j * x_spacing)
            ys.append(y)
    while len(xs) < n:
        xs.append(0.5)
        ys.append(0.5)
    return np.column_stack([np.array(xs[:n]), np.array(ys[:n])])


def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _greedy_insertion_init(grid_res=70):
    """
    Greedy insertion: add circles one by one, each at the center of the largest
    empty circle defined by the current packing and the unit-square boundary.
    Vectorized over a fine grid.
    """
    g = np.linspace(0.0, 1.0, grid_res)
    gx, gy = np.meshgrid(g, g)
    pts = []
    for _ in range(N):
        # radius at each grid point given current packing
        bnd = np.minimum.reduce([gx, 1.0 - gx, gy, 1.0 - gy])
        rad = bnd.copy()
        for (px, py) in pts:
            d = np.hypot(gx - px, gy - py)
            rad = np.minimum(rad, d)
        idx = int(np.argmax(rad))
        cx, cy = gx.ravel()[idx], gy.ravel()[idx]
        r = rad.ravel()[idx]
        if r <= 1e-9:
            # fallback
            rng = np.random.RandomState(len(pts))
            cx, cy = rng.random(), rng.random()
        pts.append((cx, cy))
    return np.array(pts)


def _simulated_annealing(pos, t_start=0.4, t_end=0.0003, n_iter=25000, seed=42, deadline=None):
    """Simulated annealing to maximize sum of radii."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    best_pos = current_pos.copy()
    best_obj = current_obj
    n = len(current_pos)
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        t = t_start * (t_end / t_start) ** (it / n_iter)
        i = rng.randint(n)
        step_size = 0.04 * t + 0.0008
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
        new_obj = _objective(new_pos)
        delta = new_obj - current_obj
        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj
    return best_pos, best_obj


def _local_search(pos, n_iter=2000, step_size=0.004, seed=42, deadline=None):
    """Local search: iteratively nudge each circle to improve the objective."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(12):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break
    return current_pos, current_obj


def _bubble_relaxation(pos, deadline, n_iter=1500, dt=0.001):
    """Physics-based bubble relaxation."""
    n = len(pos)
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_obj = float(np.sum(r_current))
    vx = np.zeros(n)
    vy = np.zeros(n)
    damping = 0.95
    for it in range(n_iter):
        if time.time() > deadline:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            x, y = current_pos[i]
            r = r_current[i]
            if x < r:
                fx[i] += (r - x) * 100.0
            if x > 1.0 - r:
                fx[i] -= (x - (1.0 - r)) * 100.0
            if y < r:
                fy[i] += (r - y) * 100.0
            if y > 1.0 - r:
                fy[i] -= (y - (1.0 - r)) * 100.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_pos[j, 0] - current_pos[i, 0]
                dy = current_pos[j, 1] - current_pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    continue
                overlap = r_current[i] + r_current[j] - dist
                if overlap > 0:
                    force_mag = overlap * 50.0
                    fx[i] -= force_mag * dx / dist
                    fy[i] -= force_mag * dy / dist
                    fx[j] += force_mag * dx / dist
                    fy[j] += force_mag * dy / dist
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        current_pos[:, 0] = np.clip(current_pos[:, 0] + vx * dt, 0.0, 1.0)
        current_pos[:, 1] = np.clip(current_pos[:, 1] + vy * dt, 0.0, 1.0)
        r_new = _max_radius_for_positions(current_pos)
        obj_new = float(np.sum(r_new))
        if obj_new > best_obj + 1e-12:
            best_pos = current_pos.copy()
            best_obj = obj_new
            r_current = r_new
    return best_pos, best_obj


def _softmin_objective_and_grad(v, n=N, beta=50.0):
    """
    Smooth surrogate over positions only: maximize sum of soft-min radii.
    v = [x0,y0,...,xn-1,yn-1] (2n values).
    Uses log-sum-exp for a smooth min of boundary and pairwise half-distances.
    Returns (energy, grad) where energy = -sum(softmin_radii).
    """
    pos = v.reshape(n, 2)
    xs, ys = pos[:, 0], pos[:, 1]

    C_bnd = np.stack([xs, ys, 1.0 - xs, 1.0 - ys], axis=1)  # (n,4)

    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0  # (n,n)

    C_all = np.hstack([C_bnd, half_dists])  # (n, 4+n)

    C_max = np.max(C_all, axis=1)  # (n,)
    exp_terms = np.exp(-beta * (C_all - C_max[:, np.newaxis]))  # (n,4+n)
    sum_exp = np.sum(exp_terms, axis=1)  # (n,)
    softmin_radii = C_max - np.log(sum_exp) / beta

    weights = exp_terms / sum_exp[:, np.newaxis]  # (n,4+n)
    grad_C = -weights  # d(softmin)/d(C_all)

    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    # Boundary: b_left=x(d/dx=1), b_bot=y(d/dy=1), b_right=1-x(d/dx=-1), b_top=1-y(d/dy=-1)
    grad_x += grad_C[:, 0] * 1.0
    grad_y += grad_C[:, 1] * 1.0
    grad_x += grad_C[:, 2] * (-1.0)
    grad_y += grad_C[:, 3] * (-1.0)

    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    coeff_x = dx / (2.0 * dists)  # diag inf -> 0
    coeff_y = dy / (2.0 * dists)
    grad_x += np.sum(grad_C[:, 4:] * coeff_x, axis=1)
    grad_y += np.sum(grad_C[:, 4:] * coeff_y, axis=1)

    energy = -np.sum(softmin_radii)
    grad = np.concatenate([grad_x, grad_y])
    return energy, grad


def _softmin_polish(pos, deadline, beta_values=None):
    """
    Polish using the smooth soft-min objective with an adaptive beta schedule
    (broad -> sharp). L-BFGS-B on positions only with exact gradient.
    """
    n = len(pos)
    current_pos = pos.copy()
    best_pos = current_pos.copy()
    best_obj = _objective(current_pos)

    if beta_values is None:
        beta_values = [8.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]

    for beta in beta_values:
        if time.time() > deadline - 1.5:
            break
        v0 = current_pos.ravel()

        def obj_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return e

        def grad_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return g

        try:
            res = minimize(
                obj_fn, v0, jac=grad_fn, method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
        except Exception:
            continue

        if time.time() > deadline - 1.5:
            break

        pos_opt = np.clip(res.x.reshape(n, 2), 0.0, 1.0)
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
        if _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj + 1e-12:
            best_pos = pos_opt
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(full_x, n, tol=1e-6):
            current_pos = pos_opt.copy()

    return best_pos, best_obj


def _soft_objective_and_grad(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, ...]. Energy = -sum(r) + mu*(boundary+overlap penalties).
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    obj = -np.sum(rs)
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)
    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    grad = np.zeros_like(x)
    grad[2::3] = -1.0
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))
    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)
    return energy, grad


def _mu_polish(pos, deadline):
    """Final high-precision polish: progressive-mu L-BFGS-B on [x,y,r] joint vars."""
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))

    mus = [800.0, 2000.0, 5000.0, 12000.0, 25000.0]
    for mu in mus:
        if time.time() > deadline - 1.0:
            break
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

        def obj_wrapper(v, m=mu):
            e, g = _soft_objective_and_grad(v, n, m)
            return e

        def grad_wrapper(v, m=mu):
            e, g = _soft_objective_and_grad(v, n, m)
            return g

        try:
            res = minimize(
                obj_wrapper, v0, jac=grad_wrapper, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1200, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
        except Exception:
            continue
        if time.time() > deadline - 1.0:
            break
        v_opt = res.x
        pos_opt = np.clip(np.column_stack([v_opt[0:n], v_opt[n:2 * n]]), 0.0, 1.0)
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))
        full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
        if _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj + 1e-12:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(full_x, n, tol=1e-6):
            current_pos = pos_opt.copy()
    return best_pos, best_r, best_obj


def _high_precision_polish(pos, deadline):
    """Combine softmin (positions) polish and progressive-mu polish."""
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))

    if time.time() < deadline - 4:
        sm_pos, sm_obj = _softmin_polish(current_pos, deadline - 3)
        if sm_obj > best_obj:
            best_pos = sm_pos
            best_r = _max_radius_for_positions(best_pos)
            best_obj = sm_obj
            current_pos = sm_pos.copy()

    if time.time() < deadline - 1.5:
        mu_pos, mu_r, mu_obj = _mu_polish(current_pos, deadline - 0.5)
        if mu_obj > best_obj:
            best_pos = mu_pos
            best_r = mu_r
            best_obj = mu_obj

    # Grow radii
    if time.time() < deadline - 0.5:
        full_x = np.concatenate([best_pos[:, 0], best_pos[:, 1], best_r])
        grown_x = _grow_radii(full_x, n, max_iter=200, step=0.0005, tol=1e-8)
        grown_pts = grown_x.reshape(n, 3)
        grown_obj = float(np.sum(grown_pts[:, 2]))
        if _is_valid(grown_x, n, tol=1e-6) and grown_obj > best_obj:
            best_pos = grown_pts[:, :2]
            best_r = grown_pts[:, 2]
            best_obj = grown_obj

    # Ultra-fine random nudges
    if time.time() < deadline - 0.1:
        rng = np.random.RandomState(42)
        for _ in range(800):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(n)
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            full_x = np.concatenate([new_pos[:, 0], new_pos[:, 1], new_r])
            if _is_valid(full_x, n, tol=1e-6) and new_obj > best_obj + 1e-12:
                best_pos = new_pos
                best_r = new_r
                best_obj = new_obj

    return best_pos, best_r, best_obj


def construct():
    global _START
    _START = time.time()
    start_time = _START
    deadline = start_time + _DEADLINE

    # Diverse initializations
    initializations = [_initial_grid()]
    initializations.append(_hex_packing_init(N))
    try:
        initializations.append(_greedy_insertion_init(grid_res=70))
    except Exception:
        pass
    for s in range(4):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: simulated annealing + local search from each init (time-budgeted)
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 55:
            break
        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=20000, seed=init_idx * 7 + 1,
            deadline=deadline - 50
        )
        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj
        if time.time() > deadline - 50:
            break
        ls_pos, ls_obj = _local_search(
            sa_pos, n_iter=1500, step_size=0.004,
            seed=init_idx * 17 + 3, deadline=deadline - 45
        )
        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: light bubble relaxation to escape shallow local minima
    if time.time() < deadline - 40:
        bubble_pos, bubble_obj = _bubble_relaxation(
            best_pos, deadline=deadline - 35, n_iter=1200, dt=0.001
        )
        if bubble_obj > best_obj:
            best_pos = bubble_pos
            best_obj = bubble_obj

    # Phase 3: gradient polish (the main quality driver)
    if time.time() < deadline - 5:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 4: final fine local search + ultra-fine nudges
    if time.time() < deadline - 2:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1500, step_size=0.0008,
            seed=12345, deadline=deadline - 1
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    if time.time() < deadline - 0.2:
        rng = np.random.RandomState(99999)
        for _ in range(500):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.00015, 0.00015)
            dy = rng.uniform(-0.00015, 0.00015)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if new_obj > best_obj + 1e-13:
                best_pos = new_pos
                best_obj = new_obj

    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
