"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _spread_objective(params, beta=50.0):
    """
    Fast spread objective: penalize close pairs using soft-min on pair distances
    plus boundary distances. Much cheaper than spectral since we avoid eigendecomposition.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    if best_c is not None:
        return best_c
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_jittered_hex():
    """Hexagonal init with small random jitter for diversity."""
    c = _initialize_hexagonal()
    r = _compute_radius(c)
    jitter = 0.3 * r
    c_jit = c + np.random.randn(N, 2) * jitter
    c_jit = np.clip(c_jit, -0.5 + 1e-12, 0.5 - 1e-12)
    return c_jit


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            maxiter = 60 if beta < 100 else 120
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    return best_c, best_r


def _get_active_pairs(c, r_target, margin=0.005):
    """Get pairs of circles that are close to violating the radius constraint."""
    tree = cKDTree(c)
    dist_limit = 2 * r_target + margin
    pairs = []
    for i in range(N):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.append((i, j))
    return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)


def _slsqp_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraints.
    Iteratively increases r_target and solves feasibility problem.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    step = 3e-5
    max_iter = 120
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        pairs = _get_active_pairs(best_c, r_target, margin=0.01)
        npairs = len(pairs)

        if npairs == 0:
            # No close pairs, just check boundary
            bd = 0.5 - np.abs(best_c)
            if np.min(bd) >= r_target:
                best_r = r_target
                step = min(step * 1.1, 1.5e-4)
                continue
            else:
                step *= 0.5
                if step < 1e-8:
                    break
                continue

        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd_flat, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            signs = -np.sign(c)
            for i in range(n):
                jac[2*i, 2*i] = signs[i, 0]
                jac[2*i+1, 2*i+1] = signs[i, 1]
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            row_indices = 2 * n + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 100, 'ftol': 1e-14, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.08, 2e-4)
            else:
                step *= 0.5
                if step < 1e-8:
                    break
        except Exception:
            break
    return best_c, best_r


def _ultra_polish(c0, r0, max_time=10.0):
    """Ultra-precise final refinement using very low temperature soft-min."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    betas = [5000, 10000, 20000, 50000, 100000]
    current_c = c0.copy()
    for beta in betas:
        if time.time() > start_time + max_time - 0.5:
            break
        def obj(params, _beta=beta):
            return _smooth_objective(params, _beta)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 80}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.49999, 0.49999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=35.0):
    """Simulated annealing with rigid-body cluster moves to escape local minima."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    T_start = 0.0015
    T_end = 1e-9
    n_steps = 8000

    def _build_clusters(c, r_target):
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return list(clusters.values())

    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 400

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        if np.random.random() < 0.65 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.004)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    return best_c, best_r


def _focused_polish(c0, r0, max_time=15.0):
    """
    Focused polish: repeatedly identify the tightest constraints and
    use L-BFGS-B to optimize only around those critical pairs.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0

    for iteration in range(30):
        if time.time() > start_time + max_time:
            break

        r_current = _compute_radius(best_c)
        if r_current <= best_r:
            break
        best_r = r_current

        # Find critical pairs (those at or near the minimum distance)
        tree = cKDTree(best_c)
        d_min = 2 * best_r
        margin = 0.002
        critical_pairs = []
        for i in range(N):
            neighbors = tree.query_ball_point(best_c[i], d_min + margin)
            for j in neighbors:
                if i < j:
                    critical_pairs.append((i, j))

        if len(critical_pairs) == 0:
            break

        # Build a focused objective that emphasizes the critical pairs
        def focused_obj(params):
            c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            boundary_dists = 0.5 - np.abs(c)
            b_min = np.min(boundary_dists)

            ci = c[np.array([p[0] for p in critical_pairs])]
            cj = c[np.array([p[1] for p in critical_pairs])]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            d_min_crit = np.min(dists) / 2.0

            # Soft min over critical pair distances and boundary
            all_vals = np.concatenate([dists / 2.0, boundary_dists.ravel()])
            r_soft = _soft_min(all_vals, 2000.0)
            return -r_soft

        try:
            result = minimize(
                focused_obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 60}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
            else:
                # Try a smaller step
                break
        except Exception:
            break

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    best_radius = 0.0
    best_c = None
    np.random.seed(42)

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Try jittered hex too
    c_jit = _initialize_jittered_hex()
    r_jit = _compute_radius(c_jit)
    if r_jit > best_radius:
        best_radius = r_jit
        best_c = c_jit.copy()

    # 3. Primary optimization with smooth objective (annealed betas)
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(best_c, max_time=min(30.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 4. Simulated annealing with rigid-body moves
    if best_c is not None and time.time() < deadline - 35:
        remaining = max(10.0, deadline - time.time() - 35)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(25.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 5. Another smooth optimization pass after SA
    if best_c is not None and time.time() < deadline - 20:
        remaining = max(5.0, deadline - time.time() - 20)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(15.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()

    # 6. Focused polish on critical pairs
    if best_c is not None and time.time() < deadline - 12:
        remaining = max(5.0, deadline - time.time() - 12)
        c_foc, r_foc = _focused_polish(best_c, best_radius, max_time=min(10.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()

    # 7. Final high-precision SLSQP polish
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(12.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    # 8. Ultra-precise final refinement
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(2.0, deadline - time.time() - 0.5)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(6.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra

    if best_c is None:
        best_c = _initialize_hexagonal()
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
