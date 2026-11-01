"""Packing 200 equal circles in a unit square using hybrid soft-min, hard-constraint, and cluster-annealing approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.optimize import NonlinearConstraint


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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances - only consider close pairs for speed
    threshold = min_dist * 1.15 
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Numerically stable soft min
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.15
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    # Combined soft min of r_pair and r_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize_soft(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with soft-min energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _slsqp_polish_hard(c0, r_target, max_time=10.0):
    """
    Hard-constrained SLSQP polish using active set of close pairs.
    Maximizes feasibility for r_target.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    # Identify active pairs: those close to the minimum distance
    dists = pdist(c0, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use a threshold to select active pairs
    threshold = max(min_dist * 1.2, 2.0 * r_target * 1.1)
    
    tree = cKDTree(c0)
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')
    
    if len(pairs) == 0:
        pairs = np.array([], dtype=int).reshape(0, 2)
        
    i_idx = pairs[:, 0] if len(pairs) > 0 else np.array([], dtype=int)
    j_idx = pairs[:, 1] if len(pairs) > 0 else np.array([], dtype=int)
    n_pairs = len(i_idx)
    
    def objective(params):
        return 0.0
    
    def grad_objective(params):
        return np.zeros(m)
    
    def pairwise_constraints(params):
        if n_pairs == 0:
            return np.array([])
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        return dists - 2.0 * r_target
    
    def grad_pairwise_constraints(params):
        if n_pairs == 0:
            return np.zeros((0, m))
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists_safe = np.where(dists > 1e-12, dists, 1e-12)
        inv_d = 1.0 / dists_safe
        
        J = np.zeros((n_pairs, m))
        J[:, i_idx * 2] += diff[:, 0] * inv_d
        J[:, i_idx * 2 + 1] += diff[:, 1] * inv_d
        J[:, j_idx * 2] -= diff[:, 0] * inv_d
        J[:, j_idx * 2 + 1] -= diff[:, 1] * inv_d
        return J
    
    def boundary_constraints(params):
        c = params.reshape(n, 2)
        x = c[:, 0]
        y = c[:, 1]
        
        left = x + 0.5 - r_target
        right = 0.5 - r_target - x
        bottom = y + 0.5 - r_target
        top = 0.5 - r_target - y
        
        return np.concatenate([left, right, bottom, top])
    
    def grad_boundary_constraints(params):
        J = np.zeros((4 * n, m))
        J[:n, np.arange(n) * 2] = 1.0
        J[n:2*n, np.arange(n) * 2] = -1.0
        J[2*n:3*n, np.arange(n) * 2 + 1] = 1.0
        J[3*n:4*n, np.arange(n) * 2 + 1] = -1.0
        return J
    
    constraints = []
    if n_pairs > 0:
        constraints.append(NonlinearConstraint(pairwise_constraints, 0.0, np.inf, 
                           jac=grad_pairwise_constraints))
    constraints.append(NonlinearConstraint(boundary_constraints, 0.0, np.inf,
                           jac=grad_boundary_constraints))
    
    bounds = [(-0.5 + r_target - 0.01, 0.5 - r_target + 0.01)] * m
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
        )
        
        c_new = np.clip(result.x.reshape(n, 2), -0.5 + r_target, 0.5 - r_target)
        r_new = _compute_radius(c_new)
        
        if r_new >= r_target - 1e-9:
            return c_new, r_new
        else:
            if r_new > _compute_radius(c0):
                return c_new, r_new
            return c0, _compute_radius(c0)
    except Exception:
        return c0, _compute_radius(c0)


def _cluster_anneal(c0, max_time=15.0, n_iter=5000):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    Moves clusters of nearby circles simultaneously to preserve local lattice structure.
    """
    start_time = time.time()
    current_c = c0.copy()
    current_r = _compute_radius(current_c)
    best_c = current_c.copy()
    best_r = current_r
    
    T_start = 0.005
    T_end = 0.0001
    n_steps = min(n_iter, int(max_time / 0.002))  # Estimate steps based on time
    
    if n_steps < 100:
        n_steps = 100
    
    for step in range(n_steps):
        if time.time() > start_time + max_time - 1:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Identify clusters using cKDTree
        tree = cKDTree(current_c)
        min_dist = 2.0 * current_r
        
        # Find pairs that are close (within 1.5x min_dist)
        close_pairs = tree.query_pairs(r=min_dist * 1.5, output_type='ndarray')
        
        if len(close_pairs) == 0:
            # No close pairs, just do single-point moves
            idx = np.random.randint(0, N)
            cluster = [idx]
        else:
            # Build clusters from close pairs using union-find or simple BFS
            # For simplicity, pick a random close pair and expand
            pair_idx = np.random.randint(0, len(close_pairs))
            i, j = close_pairs[pair_idx]
            
            # Expand cluster: add neighbors of i and j that are also close
            cluster = set([i, j])
            for k in range(N):
                if k not in cluster:
                    d_i = np.linalg.norm(current_c[k] - current_c[i])
                    d_j = np.linalg.norm(current_c[k] - current_c[j])
                    if d_i < min_dist * 1.5 or d_j < min_dist * 1.5:
                        cluster.add(k)
            
            cluster = list(cluster)
            if len(cluster) > 20:  # Cap cluster size
                cluster = np.random.choice(cluster, 20, replace=False).tolist()
        
        # Generate a random perturbation for the cluster
        # Move all points in the cluster by the same vector (rigid body)
        delta = np.random.normal(0, T, size=(len(cluster), 2))
        
        new_c = current_c.copy()
        for idx, d in zip(cluster, delta):
            new_c[idx] += d
        
        # Clip to bounds
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        # Evaluate
        new_r = _compute_radius(new_c)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r > current_r:
            current_c = new_c
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Since we're maximizing, if new_r < current_r, delta_r is negative
            # P = exp(delta_r / T) where delta_r = new_r - current_r < 0
            delta_r = new_r - current_r
            if delta_r > -1e-10:  # Essentially equal
                current_c = new_c
                current_r = new_r
            else:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_c = new_c
                    current_r = new_r
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization (Soft-min) from best initial
    if time.time() < deadline - 40:
        max_opt_time = min(15.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Cluster annealing to escape local minima
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 10
        max_anneal_time = min(15.0, remaining)
        c_anneal, r_anneal = _cluster_anneal(best_c, max_time=max_anneal_time)
        if r_anneal > best_radius:
            best_radius = r_anneal
            best_c = c_anneal.copy()
        
        # Local optimize after annealing
        if time.time() < deadline - 20:
            remaining = deadline - time.time() - 10
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: A few random restarts with soft-min
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 15
        if remaining > 5:
            max_opt_time = min(8.0, remaining)
            c_opt, r_opt = _local_optimize_soft(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Hard-constrained SLSQP Polish
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        current_c = best_c.copy()
        current_r = best_radius
        
        # Polish the current best
        if remaining > 5:
            c_pol, r_pol = _slsqp_polish_hard(current_c, current_r, max_time=min(10.0, remaining/2))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
                current_c = c_pol
                current_r = r_pol
        
        # Try to increase the radius target
        if time.time() < deadline - 10:
            bump = 1e-5
            attempts = 0
            max_attempts = 8
            
            while time.time() < deadline - 5 and attempts < max_attempts:
                target_r = current_r + bump
                c_new, r_new = _slsqp_polish_hard(current_c, target_r, max_time=4.0)
                
                if r_new > current_r:
                    current_r = r_new
                    current_c = c_new
                    if r_new > best_radius:
                        best_radius = r_new
                        best_c = c_new.copy()
                    attempts += 1
                else:
                    bump *= 0.5
                    if bump < 1e-7:
                        break
                    attempts += 1
    
    # Step 6: Final ultra-precise polish with very low temperature
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_pol, r_pol = _local_optimize_soft(best_c, max_time=min(5.0, remaining),
                                                temperature_schedule=[0.0005, 0.0002, 0.0001])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
