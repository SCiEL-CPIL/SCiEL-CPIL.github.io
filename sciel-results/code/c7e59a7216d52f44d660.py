"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
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
    shifted = np.clip(shifted, 0, 80.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = params.reshape(-1, 2)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.050, 0.080, 40):
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

    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=40.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    betas = [20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0]

    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break

        try:
            def obj(params):
                return _smooth_objective(params, beta)

            maxiter = 200 if beta < 200 else 400

            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=35.0):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0

    T_start = 0.001
    T_end = 1e-8
    n_steps = 6000

    def _build_clusters(c, r_target):
        """Group circles into clusters based on proximity."""
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
    cluster_rebuild_interval = 500

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break

        T = T_start * (T_end / T_start) ** (step / n_steps)

        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)

        if np.random.random() < 0.6 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)

            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
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


def _md_relax(c, r_target, time_limit=2.0):
    """
    Relaxes the configuration using Molecular Dynamics to resolve overlaps
    for circles of radius r_target.
    """
    c = c.copy()
    v = np.zeros_like(c)

    dt = 0.001
    damping = 0.99
    k_rep = 100.0
    k_wall = 1000.0

    start_time = time.time()

    while time.time() - start_time < time_limit:
        F = np.zeros_like(c)

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d_mat, 1.0)

        unit_vec = diff / d_mat[:, :, np.newaxis]
        overlap = np.maximum(0.0, 2 * r_target - d_mat)
        force_mat = k_rep * overlap[:, :, np.newaxis] * unit_vec
        F = np.sum(force_mat, axis=1)

        F[:, 0] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 0])
        F[:, 0] -= k_wall * np.maximum(0.0, c[:, 0] - (0.5 - r_target))
        F[:, 1] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 1])
        F[:, 1] -= k_wall * np.maximum(0.0, c[:, 1] - (0.5 - r_target))

        v = v * damping + F * dt
        c = c + v * dt
        c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)

        max_v = np.max(np.linalg.norm(v, axis=1))
        if max_v > 0.05:
            dt *= 0.5
        elif max_v < 0.01 and dt < 0.005:
            dt *= 1.5

    return c


def _md_radius_expansion(c0, r0, max_time=30.0):
    """
    Iteratively increase the radius and relax using MD.
    """
    start_time = time.time()
    c = c0.copy()
    current_r = r0

    r_max_try = 0.0395

    c = _md_relax(c, current_r, time_limit=min(5.0, max_time - 2.0))
    current_r = _compute_radius(c)

    step_size = 0.0002
    max_steps = 200

    for i in range(max_steps):
        if time.time() > start_time + max_time - 5.0:
            break

        new_r_target = current_r + step_size

        if new_r_target > r_max_try:
            break

        remaining = max(0.5, max_time - (time.time() - start_time) - 2.0)
        c = _md_relax(c, new_r_target, time_limit=min(2.0, remaining))
        actual_r = _compute_radius(c)

        if actual_r > current_r:
            current_r = actual_r
            step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break

    return c, current_r


def _slsqp_polish(c0, r0, max_time=15.0):
    """
    Final high-precision polish using SLSQP with active-set refinement.
    """
    start_time = time.time()
    c = c0.copy()
    r = r0

    def _get_active_pairs(c, r, tol=0.0001):
        """Get pairs that are near the constraint boundary."""
        d = pdist(c, metric='euclidean')
        # Find pairs where distance is close to 2r
        close_pairs = np.where(d < 2 * r + tol)[0]
        # Also include boundary constraints
        boundary_close = np.where((0.5 - np.abs(c)) < r + tol)[0]
        return close_pairs, boundary_close

    # Iterative active-set refinement with decreasing tolerance
    tolerances = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    for tol in tolerances:
        if time.time() > start_time + max_time - 3:
            break

        r_target = r

        def objective(x):
            c_local = x.reshape(-1, 2)
            # Boundary violations
            b_viol = np.maximum(0, (r_target - (0.5 - np.abs(c_local[:, 0]))))
            b_viol += np.maximum(0, (r_target - (0.5 - np.abs(c_local[:, 1]))))

            # Pairwise violations
            d = pdist(c_local, metric='euclidean')
            p_viol = np.maximum(0, 2 * r_target - d)

            return np.sum(b_viol**2) + np.sum(p_viol**2)

        try:
            result = minimize(
                objective,
                c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-18}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)

            if r_new > r:
                c = c_new
                r = r_new
        except Exception:
            pass

    return c, r


def _ultra_polish(c0, r0, max_time=10.0):
    """
    Ultra-precise final polish using very tight L-BFGS-B tolerances.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0

    # Use very high beta for near-exact minimum
    betas = [10000.0, 20000.0, 50000.0]

    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break

        try:
            def obj(params):
                return _smooth_objective(params, beta)

            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 8000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)

    best_radius = 0.0
    best_c = None

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 55:
        remaining = deadline - time.time() - 55
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 45:
        remaining = max(10.0, deadline - time.time() - 45)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 4. MD Radius Expansion
    if best_c is not None and time.time() < deadline - 25:
        remaining = max(10.0, deadline - time.time() - 25)
        c_md, r_md = _md_radius_expansion(best_c, best_radius, max_time=min(25.0, remaining))
        if r_md > best_radius:
            best_radius = r_md
            best_c = c_md.copy()

    # 5. SLSQP Polish with active-set refinement
    if best_c is not None and time.time() < deadline - 12:
        remaining = max(5.0, deadline - time.time() - 12)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(12.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final.copy()

    # 6. Ultra-precise final polish
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(2.0, deadline - time.time() - 3)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(8.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
# EVOLVE-BLOCK-END
