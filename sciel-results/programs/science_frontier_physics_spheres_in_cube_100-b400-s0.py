import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    centers = centers_flat.reshape(-1, 3)
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]
    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_sobol_init(seed=42):
    from scipy.stats import qmc
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sobol.random(N)
    centers = points - 0.5
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
    centers = np.array(points)
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _simulated_annealing(centers, deadline, iu):
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    current_r = best_r
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.95
    T = T_start
    rng = np.random.default_rng(42)
    while _time_left() > 2.0 and T > T_end:
        step_size = T * 0.1
        noise = rng.normal(0, step_size, centers.shape)
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        new_r = _compute_radius(new_centers)
        if new_r >= current_r:
            accept = True
        else:
            delta = current_r - new_r
            p = np.exp(-delta / T)
            accept = rng.random() < p
        if accept:
            centers = new_centers
            current_r = new_r
            if current_r > best_r + 1e-12:
                best_r = current_r
                best_centers = centers.copy()
        T *= alpha
    return best_centers, best_r

def _polish_repulsion(centers, deadline, iu):
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    step_size = 0.005
    decay = 0.97
    min_step = 1e-8
    margin = 1.05
    no_improve_count = 0
    max_no_improve = 150
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        forces = np.zeros_like(centers)
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        neighbor_pairs = tree.query_pairs(r=search_radius)
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        step_size *= decay
    return best_centers, best_r

def _polish_high_beta(centers, deadline, iu):
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    beta = 2000.0
    x0 = best_centers.flatten()
    try:
        res = minimize(
            _soft_min_objective,
            x0,
            args=(beta, iu),
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20}
        )
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        r = _compute_radius(new_centers)
        if r > best_r:
            best_r = r
            best_centers = new_centers
    except Exception:
        pass
    return best_centers, best_r

def _polish_very_high_beta(centers, deadline, iu):
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    betas = [5000.0, 10000.0, 50000.0]
    for beta in betas:
        if _time_left() < 3.0:
            break
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers
        except Exception:
            pass
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    start_t = time.time()
    beta_start = 50.0
    beta_end = 500.0
    n_beta_steps = 3
    for step in range(n_beta_steps):
        if _time_left() < 5.0 or (time.time() - start_t) > time_budget - 5.0:
            break
        beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        x0 = best_centers.flatten()
        try:
            max_iter = 100
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    if _time_left() > 10.0 and (time.time() - start_t) < time_budget - 5.0:
        rep_centers, rep_radius = _polish_repulsion(best_centers, deadline, iu)
        if rep_radius > best_radius:
            best_radius = rep_radius
            best_centers = rep_centers
    if _time_left() > 15.0 and (time.time() - start_t) < time_budget - 5.0:
        sa_deadline = min(deadline, start_t + time_budget - 5.0)
        sa_centers, sa_radius = _simulated_annealing(best_centers, sa_deadline, iu)
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
    if _time_left() > 3.0:
        polish_centers, polish_radius = _polish_high_beta(best_centers, deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    return best_centers, best_radius

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    iu = np.triu_indices(N, k=1)
    best_centers = None
    best_radius = 0.0

    def try_init(centers, time_budget=15.0):
        nonlocal best_centers, best_radius
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        if _time_left() > time_budget + 2.0:
            opt_centers, opt_radius = _optimize_from_initial(centers, _DEADLINE, iu, time_budget)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        else:
            if _time_left() > 3.0:
                polish_centers, polish_radius = _polish_high_beta(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers

    if _time_left() > 30.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=25.0)

    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (6, 5, 4),
        (5, 6, 4),
    ]

    for nx, ny, nz in grid_configs:
        if _time_left() < 20.0:
            break
        centers = _build_grid_init(nx, ny, nz, 0.0, seed=nx*100+ny*10+nz)
        try_init(centers, time_budget=15.0)

    if _time_left() > 15.0:
        centers = _build_sobol_init()
        try_init(centers, time_budget=12.0)

    if _time_left() > 12.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=10.0)

    if _time_left() > 8.0:
        rng = np.random.default_rng(123)
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=8.0)

    if _time_left() > 6.0:
        rng = np.random.default_rng(456)
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=6.0)

    if best_centers is not None and _time_left() > 5.0:
        rep_centers, rep_radius = _polish_repulsion(best_centers, _DEADLINE, iu)
        if rep_radius > best_radius:
            best_radius = rep_radius
            best_centers = rep_centers

        if _time_left() > 10.0:
            sa_centers, sa_radius = _simulated_annealing(best_centers, _DEADLINE, iu)
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers

        if _time_left() > 3.0:
            polish_centers, polish_radius = _polish_very_high_beta(best_centers, _DEADLINE, iu)
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)

    return best_centers
