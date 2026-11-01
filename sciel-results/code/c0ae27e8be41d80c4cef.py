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


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            maxiter = 80 if beta < 100 else 150
            
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


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    Moves clusters of nearby circles simultaneously to preserve local structure.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    
    # Temperature schedule
    T_start = 0.001
    T_end = 1e-8
    n_steps = 5000
    
    # Build cluster structure: group circles that are close together
    def _build_clusters(c, r_target):
        """Group circles into clusters based on proximity."""
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        
        # Union-Find for clustering
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
        
        # Group by cluster
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        return list(clusters.values())
    
    # Precompute clusters (recompute periodically)
    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 500
    
    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        
        # Update temperature
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Rebuild clusters periodically
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        # Choose a move: single circle or rigid cluster
        if np.random.random() < 0.6 and len(clusters) > 1:
            # Rigid body move: pick a cluster and move it
            cluster = clusters[np.random.randint(len(clusters))]
            n_cluster = len(cluster)
            # Random direction and magnitude
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            # Single circle move
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        
        # Project to square
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        
        # Accept if improvement or with probability exp(delta_r / T)
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    def _get_pairs_vectorized(c, r_target, margin=0.005):
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
    
    step = 5e-5
    max_iter = 80
    
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + step
        
        pairs = _get_pairs_vectorized(best_c, r_target, margin=0.008)
        npairs = len(pairs)
        
        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            
            if npairs == 0:
                return bd_flat
            
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
            
            if npairs == 0:
                return jac
            
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
                options={'maxiter': 150, 'ftol': 1e-14, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.05, 2e-4)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
                    
        except Exception:
            break
            
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
    if time.time() < deadline - 60:
        remaining = deadline - time.time() - 60
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 50:
        remaining = max(10.0, deadline - time.time() - 50)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 4. Another smooth optimization pass from the SA result
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(5.0, deadline - time.time() - 40)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(25.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # 5. Final high-precision polish
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(5.0, deadline - time.time() - 3)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(20.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
