"""Pack 200 equal circles in a unit square using vectorized force-directed + L-BFGS-B soft-min optimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 148  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        # Vectorized distance computation for pairs
        i_idx = np.array([p[0] for p in pairs])
        j_idx = np.array([p[1] for p in pairs])
        diffs = pts[i_idx] - pts[j_idx]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        min_pair_dist = np.min(dists)

    return min(wall_dist, min_pair_dist / 2.0)


def _soft_min(pts, beta):
    """
    Compute soft-min of (pairwise distances/2, wall distances) and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros_like(pts)

    # Wall distances: 4 per point (left, right, bottom, top)
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair distances: use KDTree to find nearby pairs
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.0)

    if len(pairs) == 0:
        pair_vals = np.array([10.0])
        pair_i = np.array([], dtype=int)
        pair_j = np.array([], dtype=int)
    else:
        pair_i = np.array([p[0] for p in pairs])
        pair_j = np.array([p[1] for p in pairs])
        diffs = pts[pair_i] - pts[pair_j]
        pair_vals = np.sqrt(np.sum(diffs ** 2, axis=1)) / 2.0

    all_vals = np.concatenate([wall_vals, pair_vals])
    n_wall = 4 * n
    n_pair = len(pair_vals)

    # Soft min computation: soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    soft_min_val = -(max_neg + np.log(sum_exp)) / beta

    # Gradient
    grad = np.zeros_like(pts)
    weights = exp_shifted / sum_exp

    # Wall gradients (vectorized)
    wall_weights = weights[:n_wall].reshape(n, 4)
    grad[:, 0] += wall_weights[:, 0] * 1.0
    grad[:, 0] += wall_weights[:, 1] * (-1.0)
    grad[:, 1] += wall_weights[:, 2] * 1.0
    grad[:, 1] += wall_weights[:, 3] * (-1.0)

    # Pair gradients
    if n_pair > 0:
        pair_weights = weights[n_wall:n_wall + n_pair]
        diffs = pts[pair_i] - pts[pair_j]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        # Avoid division by zero
        mask = dists > 1e-10
        if np.any(mask):
            unit_vecs = diffs[mask] / dists[mask, np.newaxis]
            w = pair_weights[mask]
            # Accumulate gradients
            np.add.at(grad, pair_i[mask], w[:, np.newaxis] * unit_vecs / 2.0)
            np.add.at(grad, pair_j[mask], -w[:, np.newaxis] * unit_vecs / 2.0)

    return soft_min_val, grad


def _objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B: minimize -soft_min to maximize soft_min."""
    pts = x.reshape(-1, 2)
    val, grad = _soft_min(pts, beta)
    # We want to maximize soft_min, so minimize -soft_min
    return -val, -grad.reshape(-1)


def _generate_hex_initial(spacing, margin=0.02, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal lattice of candidate points inside [-0.5-margin, 0.5+margin]^2."""
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing

    pts = []
    y = -0.5 - margin + offset_y
    row_idx = 0
    while y <= 0.5 + margin:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 - margin + shift + offset_x
        while x <= 0.5 + margin:
            pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1

    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def _force_directed_phase(pts, max_iters=50000, lr_start=0.005, cooling=0.99995):
    """
    Vectorized force-directed optimization to spread out points.
    Points repel each other and are attracted to center.
    """
    pts = pts.copy()
    n = len(pts)
    lr = lr_start

    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 30:
            break

        # Compute pairwise forces (repulsion)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=0.5)

        if len(pairs) > 0:
            pair_i = np.array([p[0] for p in pairs])
            pair_j = np.array([p[1] for p in pairs])
            diffs = pts[pair_i] - pts[pair_j]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            mask = dists > 1e-10
            if np.any(mask):
                # Force proportional to 1/d^2 (Coulomb-like)
                forces = diffs[mask] / (dists[mask, np.newaxis] ** 3)
                np.add.at(pts, pair_i[mask], lr * forces)
                np.add.at(pts, pair_j[mask], -lr * forces)

        # Wall attraction (keep points away from walls)
        wall_force = np.zeros_like(pts)
        wall_force[:, 0] += np.maximum(0, -0.45 - pts[:, 0])  # left wall
        wall_force[:, 0] -= np.maximum(0, pts[:, 0] - 0.45)   # right wall
        wall_force[:, 1] += np.maximum(0, -0.45 - pts[:, 1])  # bottom wall
        wall_force[:, 1] -= np.maximum(0, pts[:, 1] - 0.45)   # top wall
        pts += lr * wall_force * 0.1

        # Clip to square
        pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
        pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)

        lr *= cooling

    return pts


def _projected_gradient_phase(pts, max_iters=10000, beta_start=50.0, beta_end=500.0, lr_start=0.002, cooling=0.9999):
    """
    Projected gradient descent on soft-min objective with increasing beta.
    """
    pts = pts.copy()
    n = len(pts)
    lr = lr_start

    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 20:
            break

        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress

        val, grad = _soft_min(pts, beta)

        pts += lr * grad
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

        lr *= cooling

        if np.max(np.abs(grad)) < 1e-10:
            break

    return pts


def _lbfgs_polish(pts, beta_start=50.0, beta_end=5000.0, max_time=30.0):
    """
    Use L-BFGS-B to optimize the soft-min objective with increasing beta.
    """
    pts = pts.copy()
    t0 = time.time()
    n = len(pts)
    x0 = pts.flatten()

    # Bounds: keep points inside the square
    bounds = [(-0.5, 0.5)] * (2 * n)

    # Phase 1: Lower beta for broad optimization
    beta1 = beta_start
    if time.time() - t0 < max_time - 10:
        res1 = minimize(
            _objective_and_grad,
            x0,
            args=(beta1,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res1.success or res1.fun < 0:
            pts = res1.x.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    # Phase 2: Higher beta for sharper min
    beta2 = beta_end
    if time.time() - t0 < max_time - 5:
        x1 = pts.flatten()
        res2 = minimize(
            _objective_and_grad,
            x1,
            args=(beta2,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res2.success or res2.fun < 0:
            pts = res2.x.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    return pts


def _generate_jittered_hex(spacing, margin=0.02, jitter=0.01, offset_x=0.0, offset_y=0.0, seed=42):
    """Generate a hexagonal lattice with random jitter."""
    np.random.seed(seed)
    pts = _generate_hex_initial(spacing, margin, offset_x, offset_y)
    if len(pts) == 0:
        return pts
    pts = pts + np.random.uniform(-jitter, jitter, size=pts.shape)
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    return pts


def _perturb_and_optimize(pts, num_perturb=5, seed=42):
    """
    Try multiple perturbations and keep the best result.
    """
    np.random.seed(seed)
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    for i in range(num_perturb):
        if time.time() > _TIME_LIMIT - 40:
            break

        # Perturb points
        perturbed = pts + np.random.uniform(-0.02, 0.02, size=pts.shape)
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.5, 0.5)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.5, 0.5)

        # Quick force-directed phase
        polished = _force_directed_phase(perturbed, max_iters=5000)
        r = _get_radius(polished)

        if r > best_r:
            best_r = r
            best_pts = polished.copy()

    return best_pts, best_r


def _run_gradient_opt(pts, time_budget=60.0):
    """
    Run the full gradient optimization pipeline.
    """
    t0 = time.time()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    # Phase 1: Force-directed
    if time.time() - t0 < time_budget - 20:
        fd_pts = _force_directed_phase(pts, max_iters=20000)
        r_fd = _get_radius(fd_pts)
        if r_fd > best_r:
            best_r = r_fd
            best_pts = fd_pts

    # Phase 2: Projected gradient
    if time.time() - t0 < time_budget - 15:
        pg_pts = _projected_gradient_phase(best_pts, max_iters=5000)
        r_pg = _get_radius(pg_pts)
        if r_pg > best_r:
            best_r = r_pg
            best_pts = pg_pts

    # Phase 3: L-BFGS-B polish
    if time.time() - t0 < time_budget - 5:
        lb_pts = _lbfgs_polish(best_pts, beta_start=50.0, beta_end=5000.0,
                               max_time=min(30.0, time_budget - 5 - (time.time() - t0)))
        r_lb = _get_radius(lb_pts)
        if r_lb > best_r:
            best_r = r_lb
            best_pts = lb_pts

    return best_pts, best_r


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Strategy:
    # 1. Try multiple hex lattice configurations with different spacings and offsets
    # 2. Run gradient optimization on each
    # 3. Keep the best result

    # Different lattice spacings and offsets to try
    configs = [
        (0.065, 0.0, 0.0, 0.0),
        (0.070, 0.0, 0.0, 0.0),
        (0.075, 0.0, 0.0, 0.0),
        (0.068, 0.02, 0.01, 0.0),
        (0.072, 0.02, 0.01, 0.0),
        (0.070, 0.03, 0.02, 0.0),
    ]

    for idx, (spacing, ox, oy, seed) in enumerate(configs):
        if time.time() - t0 > _TIME_LIMIT - 50:
            break

        # Generate hex lattice
        lattice = _generate_hex_initial(spacing, margin=0.03, offset_x=ox, offset_y=oy)

        if len(lattice) < N:
            continue

        # Select N points (take first N or use a simple selection)
        if len(lattice) <= N:
            selected = lattice.copy()
        else:
            # Select points that are well-spread
            # Use a simple greedy selection
            selected = []
            remaining = lattice.copy()
            for _ in range(N):
                if len(remaining) == 0:
                    break
                # Pick the point farthest from already selected
                if len(selected) == 0:
                    # Pick the most central point
                    wall_dists = np.minimum(
                        np.minimum(remaining[:, 0] + 0.5, 0.5 - remaining[:, 0]),
                        np.minimum(remaining[:, 1] + 0.5, 0.5 - remaining[:, 1])
                    )
                    best_idx = int(np.argmax(wall_dists))
                else:
                    sel_arr = np.array(selected)
                    tree = cKDTree(sel_arr)
                    dists, _ = tree.query(remaining)
                    best_idx = int(np.argmax(dists))
                selected.append(remaining[best_idx])
                remaining = np.delete(remaining, best_idx, axis=0)
            selected = np.array(selected)

        if len(selected) < N:
            continue

        # Run optimization
        time_budget = min(20.0, _TIME_LIMIT - 50 - (time.time() - t0))
        if time_budget > 5.0:
            optimized, r_opt = _run_gradient_opt(selected, time_budget=time_budget)
        else:
            optimized = selected
            r_opt = _get_radius(selected)

        if r_opt > best_r:
            best_r = r_opt
            best_pts = optimized.copy()

    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 15:
        final_polish = _lbfgs_polish(best_pts, beta_start=100.0, beta_end=10000.0,
                                     max_time=min(20.0, _TIME_LIMIT - 15 - (time.time() - t0)))
        r_final = _get_radius(final_polish)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_polish.copy()

    # Fallback
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]

    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
