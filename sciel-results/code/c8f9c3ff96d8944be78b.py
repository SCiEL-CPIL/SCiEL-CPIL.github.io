"""Optimized packing of 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.5
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _soft_min_value_and_grad(pts, t):
    """
    Compute soft-min value and gradient of all constraints (pair distances/2 and wall distances).
    Vectorized for speed.
    """
    pts = np.array(pts, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)

    # Wall distances
    max_coord = np.abs(pts).max(axis=1)
    wall_dists = 0.5 - max_coord

    # Build constraint vector: pair distances/2 (upper triangle) + wall distances
    triu_i, triu_j = np.triu_indices(n, k=1)
    pair_constraints = dist[triu_i, triu_j] / 2.0
    all_constraints = np.concatenate([pair_constraints, wall_dists])

    # Soft-min via log-sum-exp
    neg_c = -all_constraints / t
    neg_c_clipped = np.clip(neg_c, -500, 500)
    max_neg_c = np.max(neg_c_clipped)
    exp_neg_c = np.exp(neg_c_clipped - max_neg_c)
    sum_exp = np.sum(exp_neg_c)
    log_sum_exp = max_neg_c + np.log(sum_exp)
    smooth_min = -t * log_sum_exp

    # Weights for gradient
    weights = exp_neg_c / sum_exp

    # Gradient computation (vectorized)
    grad = np.zeros_like(pts)

    # Pair contribution: d/dp_i [ -t * log(sum exp(-c_k/t)) ] = sum_k w_k * (1/t) * dc_k/dp_i
    # For pair (i,j): c = dist_ij/2, dc/dp_i = (p_i - p_j)/dist_ij, dc/dp_j = (p_j - p_i)/dist_ij
    pair_weights = weights[:len(pair_constraints)]
    d_vecs = pts[triu_i] - pts[triu_j]
    dists = np.linalg.norm(d_vecs, axis=1)
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    unit_vecs = d_vecs / dists_safe[:, np.newaxis]

    # grad[i] += -w * (1/t) * (p_i - p_j)/dist_ij * 1/2 ... wait, let me redo this
    # c_k = dist_ij / 2
    # d c_k / d p_i = (p_i - p_j) / dist_ij
    # d c_k / d p_j = (p_j - p_i) / dist_ij
    # d(smooth_min)/d p_i = sum_k w_k * (1/t) * d c_k / d p_i ... no
    # smooth_min = -t * log(sum exp(-c_k/t))
    # d(smooth_min)/d p_i = -t * (1/sum) * sum_k exp(-c_k/t) * (-1/t) * d c_k/d p_i
    # = sum_k w_k * d c_k/d p_i
    # So grad[i] += sum over pairs involving i of w_k * (p_i - p_j)/dist_ij

    # Accumulate pair gradients
    for idx in range(len(triu_i)):
        i = triu_i[idx]
        j = triu_j[idx]
        w = pair_weights[idx]
        if w < 1e-15:
            continue
        grad[i] += w * unit_vecs[idx]
        grad[j] -= w * unit_vecs[idx]

    # Wall contribution: c = 0.5 - max(|x|,|y|,|z|)
    # d c / d p_k = -sign(p_k) if p_k is the max absolute coordinate
    wall_weights = weights[len(pair_constraints):]
    for idx in range(n):
        w = wall_weights[idx]
        if w < 1e-15:
            continue
        p = pts[idx]
        abs_p = np.abs(p)
        max_idx = np.argmax(abs_p)
        if abs_p[max_idx] > 1e-10:
            grad[idx, max_idx] += -w * np.sign(p[max_idx])

    return smooth_min, grad


def _soft_min_objective(pts, t):
    """Objective: negative of soft-min (minimize to maximize soft-min)."""
    val, _ = _soft_min_value_and_grad(pts, t)
    return -val


def _soft_min_gradient(pts, t):
    """Gradient of the soft-min objective."""
    _, g = _soft_min_value_and_grad(pts, t)
    return -g


def _soft_min_optimize(pts, deadline, t_schedule=None):
    """
    Optimize using L-BFGS-B with adaptive temperature schedule.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)

    if t_schedule is None:
        t_schedule = [0.02, 0.01, 0.005, 0.002, 0.001]

    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (n * 3)

    for t in t_schedule:
        if time.time() > deadline - 5:
            break

        def objective(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            return _soft_min_objective(p, t)

        def gradient(x):
            p = x.reshape(-1, 3)
            p = np.clip(p, -0.5, 0.5)
            g = _soft_min_gradient(p, t)
            return g.flatten()

        try:
            res = minimize(
                objective, x0, method='L-BFGS-B', jac=gradient,
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )

            new_pts = res.x.reshape(-1, 3)
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_r = _compute_radius(new_pts)

            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.flatten()

        except Exception:
            pass

    return best_pts, best_r


def _grid_init(n):
    """Generate a grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    best_pts = None
    best_r = 0

    for m1 in range(max(1, m - 2), m + 3):
        for m2 in range(max(1, m - 2), m + 3):
            m3 = max(1, int(round(n / (m1 * m2))))
            if m1 * m2 * m3 < n:
                continue

            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5

            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            if len(grid) >= n:
                subset = grid[:n]
                r = _compute_radius(subset)
                if r > best_r:
                    best_r = r
                    best_pts = subset

    if best_pts is None:
        m = int(np.ceil(n ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:n]
        best_r = _compute_radius(best_pts)

    return best_pts, best_r


def _bcc_init(n):
    """Generate BCC lattice initialization."""
    best_pts = None
    best_r = 0

    for a in np.linspace(0.22, 0.38, 25):
        n_cells = int(1.0 / a)
        pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        pts.append([x, y, z])
                    x2 = -0.5 + (i + 0.5) * a
                    y2 = -0.5 + (j + 0.5) * a
                    z2 = -0.5 + (k + 0.5) * a
                    if -0.5 <= x2 <= 0.5 and -0.5 <= y2 <= 0.5 and -0.5 <= z2 <= 0.5:
                        pts.append([x2, y2, z2])

        pts = np.array(pts)
        if len(pts) < n:
            continue

        np.random.seed(42)
        for trial in range(20):
            if trial == 0:
                subset = pts[:n]
            else:
                idx = np.random.choice(len(pts), n, replace=False)
                subset = pts[idx]

            r = _compute_radius(subset)
            if r > best_r:
                best_r = r
                best_pts = subset

    if best_pts is None:
        pts, r = _grid_init(n)
        return pts, r

    return best_pts, best_r


def _random_init(n, seed=None):
    """Generate random initialization."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(n, 3))
    r = _compute_radius(pts)
    return pts, r


def _sgld_optimize(pts, deadline, seed=None):
    """
    Stochastic Gradient Langevin Dynamics with adaptive annealing.
    """
    if seed is not None:
        np.random.seed(seed)

    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)

    t0 = 0.05
    t_min = 0.001
    lr0 = 0.01
    n_steps = 500

    for step in range(n_steps):
        if time.time() > deadline - 10:
            break

        alpha = 1.0 - step / n_steps
        t = t_min + (t0 - t_min) * alpha
        lr = lr0 * alpha

        _, grad = _soft_min_value_and_grad(pts, t)
        grad = -grad  # gradient of objective = -gradient of soft-min

        noise = np.random.randn(*pts.shape) * np.sqrt(2 * t * lr)
        pts = pts - lr * grad + noise
        pts = np.clip(pts, -0.5, 0.5)

        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts, best_r


def _local_polish(pts, deadline):
    """Final high-precision polish using coordinate-wise moves on the exact objective."""
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()

    lr = 0.0005
    for it in range(300):
        if time.time() > deadline - 2:
            break

        # Find the most violated constraint
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)

        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_d = dist[i, j]

        wall_dists = 0.5 - np.abs(pts).max(axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_d = wall_dists[min_wall_idx]

        if min_d / 2.0 < min_wall_d:
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            pts[i] += lr * direction
            pts[j] -= lr * direction
        else:
            p = pts[min_wall_idx]
            max_coord_idx = np.argmax(np.abs(p))
            if p[max_coord_idx] > 0:
                pts[min_wall_idx, max_coord_idx] -= lr
            else:
                pts[min_wall_idx, max_coord_idx] += lr

        pts = np.clip(pts, -0.5, 0.5)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        else:
            pts = best_pts.copy()
            lr *= 0.7

    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Multiple initializations
    inits = []

    pts_grid, r_grid = _grid_init(N)
    inits.append((pts_grid, r_grid))

    pts_bcc, r_bcc = _bcc_init(N)
    inits.append((pts_bcc, r_bcc))

    for seed in [42, 123, 456, 789]:
        pts_rand, r_rand = _random_init(N, seed=seed)
        inits.append((pts_rand, r_rand))

    inits.sort(key=lambda x: x[1], reverse=True)

    best_pts = inits[0][0]
    best_r = inits[0][1]

    # Phase 1: Soft-min L-BFGS-B optimization from each good init
    for init_idx, (init_pts, init_r) in enumerate(inits):
        if time.time() > deadline - 20:
            break

        if init_r < best_r * 0.85 and init_idx > 0:
            continue

        try:
            opt_pts, opt_r = _soft_min_optimize(init_pts, deadline)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
        except Exception:
            continue

    # Phase 2: SGLD to escape local minima
    if time.time() < deadline - 15:
        try:
            sgld_pts, sgld_r = _sgld_optimize(best_pts, deadline, seed=42)
            if sgld_r > best_r:
                best_r = sgld_r
                best_pts = sgld_pts.copy()
        except Exception:
            pass

    # Phase 3: Another SGLD with different seed
    if time.time() < deadline - 10:
        try:
            sgld_pts2, sgld_r2 = _sgld_optimize(best_pts, deadline, seed=123)
            if sgld_r2 > best_r:
                best_r = sgld_r2
                best_pts = sgld_pts2.copy()
        except Exception:
            pass

    # Phase 4: Final soft-min polish
    if time.time() < deadline - 5:
        try:
            polished_pts, polished_r = _soft_min_optimize(best_pts, deadline, t_schedule=[0.001, 0.0005])
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
        except Exception:
            pass

    # Phase 5: Exact local polish
    if time.time() < deadline - 2:
        polished_pts, polished_r = _local_polish(best_pts, deadline)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
# EVOLVE-BLOCK-END
