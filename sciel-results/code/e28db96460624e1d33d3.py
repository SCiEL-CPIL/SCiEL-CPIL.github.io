"""Packing 200 equal circles in a unit square: Multi-temperature L-BFGS-B + QP Polish + Delaunay Springs."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _fast_score(pts):
    """Faster score using KDTree for pairwise distances."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])
    r = min(min_boundary, min_pair / 2.0)
    return r if r > 0 else 0.0

def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization with multiple aspect ratios."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 25):
        for aspect in [1.0, 0.95, 1.05, 0.9, 1.1]:
            pts_list = []
            row = 0
            while True:
                y = -0.5 + s/2 + row * s * np.sqrt(3) / 2 * aspect
                if y > 0.5 - s/2:
                    break
                offset = (row % 2) * s / 2
                x = -0.5 + s/2 + offset
                while x <= 0.5 - s/2:
                    pts_list.append((x, y))
                    x += s
                row += 1
            if len(pts_list) < N:
                continue
            pts = np.array(pts_list[:N])
            if jitter_scale > 0:
                pts += rng.normal(0, jitter_scale, size=pts.shape)
                pts = np.clip(pts, -0.49, 0.49)
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _random_init(seed=0, n_clusters=10):
    """Generate a random clustered initialization for diversity."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    cluster_centers = rng.uniform(-0.4, 0.4, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        pts[i] = pts[i] * 0.5 + cluster_centers[c] * 0.5
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _delaunay_spring_relax(pts, time_budget):
    """
    Delaunay-topology spring relaxation: maintain local hexagonal geometry
    via spring forces on Delaunay edges while pushing points toward boundaries.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _score(pts)
    best_pts = pts.copy()
    best_r = current_r

    lr = 0.004
    spring_k = 1.0
    pressure_k = 0.3
    target_spacing = 1.0 / np.sqrt(N) * 1.15

    while time.time() - t0 < time_budget:
        # Build Delaunay triangulation
        try:
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edges.add((int(simplex[i]), int(simplex[j])))
        except Exception:
            break

        forces = np.zeros_like(pts)

        # Spring forces: push pairs toward target spacing
        for i, j in edges:
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            # Spring force: (d - target) * direction
            f = spring_k * (d - target_spacing) / d
            forces[i] += f * diff
            forces[j] -= f * diff

        # Pressure: push points toward boundaries (outward)
        # Points near center get pushed outward to fill space
        center_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        outward = pts / np.maximum(center_dist, 0.1)[:, np.newaxis]
        forces += pressure_k * outward * (1.0 - center_dist / 0.5)

        # Boundary constraint: push inward if too close
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        r_est = max(current_r, 0.001)
        close = boundary_dists < r_est * 1.2
        if np.any(close):
            forces[close] -= pts[close] * 2.0

        # Normalize and update
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = 0.3
        too_large = force_norms[:, 0] > max_norm
        if np.any(too_large):
            forces[too_large] *= (max_norm / force_norms[too_large, 0])[:, np.newaxis]

        pts += lr * forces
        pts = np.clip(pts, -0.499, 0.499)

        new_r = _fast_score(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
            lr *= 0.998
        else:
            lr *= 0.98

        if lr < 1e-6:
            break

    return best_pts

def _qp_optimize(points, time_budget):
    """
    QP-style constrained gradient descent with adaptive radius.
    Directly targets the min-distance objective.
    """
    t0 = time.time()
    pts = points.copy()
    r = _score(pts)
    best_pts = pts.copy()
    best_r = r

    lr_init = 0.004
    lr_decay = 0.9995
    lr_min = 1e-5
    r_step = 0.00005

    lr = lr_init

    while time.time() - t0 < time_budget:
        # Compute pairwise distances via KDTree for speed
        tree = cKDTree(pts)
        # Use k=10 to capture enough neighbors
        dists, indices = tree.query(pts, k=10)

        # Build gradient from pairwise violations
        forces = np.zeros_like(pts)
        for i in range(N):
            for j_idx in range(1, 10):
                j = int(indices[i, j_idx])
                d = dists[i, j_idx]
                if d < 2.0 * r and d > 1e-10:
                    viol = 2.0 * r - d
                    weight = viol / d
                    diff = pts[i] - pts[j]
                    forces[i] += weight * diff / d

        # Boundary forces
        bx = 0.5 - np.abs(pts[:, 0])
        by = 0.5 - np.abs(pts[:, 1])
        active_x = bx <= by
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        forces[:, 0] -= viol_x * np.sign(pts[:, 0])
        forces[:, 1] -= viol_y * np.sign(pts[:, 1])

        # Update
        pts += lr * forces
        pts = np.clip(pts, -0.4999, 0.4999)

        current_r = _fast_score(pts)
        if current_r >= r:
            r = min(current_r + r_step * 0.3, current_r + 1e-6)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            r = max(current_r, r - r_step * 0.5)

        lr *= lr_decay
        if lr < lr_min:
            break

    return best_pts

def _soft_min_neg(x, alpha, tree_cache=None):
    """
    Objective function for L-BFGS-B using numerically stable log-sum-exp.
    Uses KDTree for pairwise distances to reduce cost.
    """
    pts = x.reshape(N, 2)

    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()

    # Pairwise distances via KDTree (k=8 for speed)
    try:
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=8)
        pairwise_dists = dists[:, 1:].flatten()
    except Exception:
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        pairwise_dists = dist[~np.isinf(dist)]

    all_dists = np.concatenate([boundary_dists, pairwise_dists])

    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha

    return -soft_min_val

def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature annealing schedule.
    """
    t0 = time.time()
    x0 = pts.flatten()

    phases = [
        {"alpha": 8.0, "max_iter": 150, "gtol": 1e-3},
        {"alpha": 25.0, "max_iter": 200, "gtol": 1e-5},
        {"alpha": 80.0, "max_iter": 250, "gtol": 1e-6},
        {"alpha": 250.0, "max_iter": 250, "gtol": 1e-7},
        {"alpha": 800.0, "max_iter": 200, "gtol": 1e-8},
        {"alpha": 2000.0, "max_iter": 150, "gtol": 1e-9},
    ]

    for phase in phases:
        if time.time() - t0 > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass

    return x0.reshape(N, 2)

def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)

    rng = np.random.default_rng(54321)
    step_size = 0.002
    min_step = 1e-7
    stagnation_count = 0

    while time.time() - t0 < time_budget:
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)

        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)

        current_r = min(min_boundary, min_pair / 2.0)

        active_idx = set()
        tight_boundary = np.where(boundary_dists < current_r * 1.05)[0]
        active_idx.update(tight_boundary.tolist())

        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=5)
        for i in range(N):
            for j in range(1, 5):
                if dists[i, j] < 2.0 * current_r * 1.05:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))

        if not active_idx:
            break

        active_list = list(active_idx)
        n_active = len(active_list)
        improved = False

        # Strategy 1: Random perturbations
        for _ in range(15):
            perturb = np.zeros((N, 2))
            perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
            trial = best_pts + perturb
            trial = np.clip(trial, -0.4999, 0.4999)
            trial_r = _score(trial)
            if trial_r > best_r:
                best_pts = trial
                best_r = trial_r
                improved = True
                break

        # Strategy 2: Directed perturbations
        if not improved:
            for i in active_list:
                nearest_dist, nearest_idx = tree.query(best_pts[i], k=2)
                for j_idx in range(1, 2):
                    j = int(nearest_idx[j_idx])
                    if j == i:
                        continue
                    diff = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        perturb = np.zeros((N, 2))
                        perturb[i] = diff / norm * step_size
                        trial = best_pts + perturb
                        trial = np.clip(trial, -0.4999, 0.4999)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                if improved:
                    break

        if improved:
            step_size *= 0.95
            stagnation_count = 0
        else:
            step_size *= 0.85
            stagnation_count += 1
            if step_size < min_step or stagnation_count > 50:
                break

    return best_pts

def _high_precision_polish(pts, time_budget):
    """
    Very high-precision polish using a smooth surrogate with very high alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()

    phases = [
        {"alpha": 5000.0, "max_iter": 100, "gtol": 1e-10},
        {"alpha": 15000.0, "max_iter": 80, "gtol": 1e-11},
    ]

    for phase in phases:
        if time.time() - t0 > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           bounds=[(-0.49999, 0.49999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-14, 'maxls': 30})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass

    return x0.reshape(N, 2)

def construct():
    t_start = time.time()

    # 1. Initialization - try diverse starting points
    best_pts = None
    best_r = 0.0

    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.003, seed=1),
        _hex_init(jitter_scale=0.008, seed=2),
        _hex_init(jitter_scale=0.015, seed=3),
        _random_init(seed=42),
        _random_init(seed=123),
    ]

    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # 2. Delaunay spring relaxation (new: maintains hex topology while adapting to boundaries)
    remaining = DEADLINE - time.time()
    if remaining > 15:
        ds_budget = min(remaining * 0.15, 12.0)
        ds_pts = _delaunay_spring_relax(best_pts, ds_budget)
        ds_r = _score(ds_pts)
        if ds_r > best_r:
            best_r = ds_r
            best_pts = ds_pts.copy()

    # 3. QP-style constrained gradient descent (from inspiration: directly targets min-distance)
    remaining = DEADLINE - time.time()
    if remaining > 20:
        qp_budget = min(remaining * 0.2, 18.0)
        qp_pts = _qp_optimize(best_pts, qp_budget)
        qp_r = _score(qp_pts)
        if qp_r > best_r:
            best_r = qp_r
            best_pts = qp_pts.copy()

    # 4. L-BFGS-B Polish with multi-temperature annealing
    remaining = DEADLINE - time.time()
    if remaining > 25:
        lbfgs_budget = min(remaining * 0.35, 45.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # 5. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 10:
        polish_budget = min(remaining * 0.2, 15.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 6. Second L-BFGS-B pass
    remaining = DEADLINE - time.time()
    if remaining > 20:
        lbfgs_budget = min(remaining * 0.3, 30.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # 7. Active Set Polish again
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.25, 12.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 8. High-precision polish
    remaining = DEADLINE - time.time()
    if remaining > 10:
        hp_budget = min(remaining * 0.35, 20.0)
        hp_pts = _high_precision_polish(best_pts, hp_budget)
        hp_pts = np.clip(hp_pts, -0.49999, 0.49999)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()

    # 9. Final active set polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.5, 8.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # Final safety check
    if best_pts is None:
        best_pts = _hex_init()

    best_pts = np.clip(best_pts, -0.49999, 0.49999)

    return best_pts
