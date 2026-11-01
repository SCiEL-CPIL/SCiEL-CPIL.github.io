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


def _energy_focused(params, temperature=0.02):
    """Focused energy: only considers pairs within a threshold of the current minimum."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    if N < 2:
        return -b_min
    dists = pdist(c, metric='euclidean')
    d_min = np.min(dists)
    T = max(temperature, 1e-8)
    threshold = d_min * 1.15 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = d_min
    b_threshold = b_min * 1.15 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
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


def _local_optimize_focused(c0, max_time=20.0, temps=None):
    """Local optimization using the focused energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if temps is None:
        temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _md_relaxation(c0, r0, max_time=40.0):
    """
    Rigid-body dynamics relaxation with soft-contact model.
    Iteratively increases target radius and relaxes the system.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    n = N
    
    # Parameters for the soft-contact potential
    # Force ~ (2r - d)^power if d < 2r
    power = 4.0 
    stiffness = 100.0
    damping = 0.98
    dt = 0.005
    max_iter_per_step = 200
    
    # Initial radius increment
    dr = 0.0005
    min_dr = 1e-5

    while time.time() < start_time + max_time:
        # Increase target radius
        target_r = current_r + dr
        
        # Check if we can even fit this radius in the square
        if target_r > 0.5:
            break
            
        # Run MD relaxation for this target_r
        velocities = np.zeros((n, 2))
        settled = False
        
        for it in range(max_iter_per_step):
            if time.time() > start_time + max_time:
                break
                
            # Calculate forces
            forces = np.zeros((n, 2))
            
            # Pairwise interactions
            # Use KDTree for efficiency if N is large, but for N=200 O(N^2) is ~40k ops, very fast.
            # Let's stick to vectorized O(N^2) for simplicity and speed in Python loop overhead.
            # Actually, vectorized O(N^2) in NumPy is fast enough for 200 points.
            diffs = current_c[:, np.newaxis, :] - current_c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            
            # Avoid division by zero
            np.fill_diagonal(dists, 1.0)
            
            # Overlap condition: dist < 2 * target_r
            overlap = dists < 2.0 * target_r
            # Exclude self-interaction
            np.fill_diagonal(overlap, False)
            
            # Calculate repulsive force magnitude
            # F = stiffness * (2r - d)^power
            # Direction: unit vector from j to i
            # We need to be careful with the gradient.
            # Potential V = stiffness * (2r - d)^(power+1) / (power+1)
            # Force F = -grad V = stiffness * (2r - d)^power * (d_vec / d)
            
            # Only compute for overlapping pairs
            # We can use masking
            # For simplicity and speed in this specific N, we can just compute all and mask
            # But let's optimize slightly:
            # Create a matrix of distances, mask out non-overlaps and diagonals
            valid_dists = np.where(overlap, dists, 1.0) # Safe value for non-overlap
            
            # Calculate penetration depth
            pen = np.maximum(0, 2.0 * target_r - dists)
            # Force magnitude
            # F_mag = stiffness * pen^power
            f_mag = stiffness * (pen ** power)
            
            # Force vector
            # f_vec = f_mag * (diff / dist)
            # Note: diffs[i,j] is c_i - c_j. Force on i from j should be repulsive (away from j).
            # So direction is c_i - c_j.
            # f_mag is zero if no overlap.
            
            # Handle zero distance (coincident points) - randomize slightly? 
            # For now, assume dists > 0 due to previous steps.
            safe_dists = np.where(dists < 1e-8, 1e-8, dists)
            
            # Force on i from j:
            # F_ij = f_mag[i,j] * (c_i - c_j) / dist[i,j]
            # Sum over j for each i.
            
            # Vectorized sum:
            # forces[i] = sum_j ( f_mag[i,j] * diffs[i,j] / safe_dists[i,j] )
            # diffs is (N, N, 2). f_mag is (N, N).
            # We need to scale diffs by f_mag / safe_dists along the last axis.
            
            scale = f_mag / safe_dists
            # scale is (N, N)
            # force contribution from j to i: scale[i,j] * diffs[i,j]
            # Sum over j (axis 1)
            forces += np.sum(scale[:, :, np.newaxis] * diffs, axis=1)
            
            # Wall interactions
            # Distance to walls
            # Left: x - (-0.5) = x + 0.5
            # Right: 0.5 - x
            # Bottom: y + 0.5
            # Top: 0.5 - y
            
            d_left = current_c[:, 0] + 0.5
            d_right = 0.5 - current_c[:, 0]
            d_bottom = current_c[:, 1] + 0.5
            d_top = 0.5 - current_c[:, 1]
            
            # Penetration
            pen_left = np.maximum(0, target_r - d_left)
            pen_right = np.maximum(0, target_r - d_right)
            pen_bottom = np.maximum(0, target_r - d_bottom)
            pen_top = np.maximum(0, target_r - d_top)
            
            # Forces (pointing inward)
            f_left = stiffness * (pen_left ** power)
            f_right = stiffness * (pen_right ** power)
            f_bottom = stiffness * (pen_bottom ** power)
            f_top = stiffness * (pen_top ** power)
            
            forces[:, 0] += f_left - f_right
            forces[:, 1] += f_bottom - f_top
            
            # Update velocities (Symplectic Euler)
            velocities += forces * dt
            velocities *= damping
            
            # Update positions
            current_c += velocities * dt
            
            # Check for settling
            # If max velocity is small, we can stop early
            if np.max(np.abs(velocities)) < 1e-5:
                settled = True
                break
        
        # Evaluate new radius
        # Clip to square
        current_c = np.clip(current_c, -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(current_c)
        
        if r_new > best_r:
            best_r = r_new
            best_c = current_c.copy()
            current_r = r_new
            # Increase step size if successful? Or keep same?
            # Usually keep same or increase slightly.
            # Let's try increasing dr slightly to speed up convergence
            dr = min(dr * 1.05, 0.002)
        else:
            # If we didn't improve, reduce step size
            dr *= 0.8
            if dr < min_dr:
                break
            # Revert to best known good configuration? 
            # Or keep current if it's close?
            # To avoid getting stuck in bad configurations, revert to best_c if current is worse than best_r significantly
            if r_new < best_r - 1e-5:
                 current_c = best_c.copy()
            # If we can't even fit the current_r (due to numerical issues), break
            if r_new < current_r - 1e-4:
                 break

    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """High-precision polish using SLSQP with active constraints."""
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

    step = 3e-5
    max_iter = 120
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        pairs = _get_pairs_vectorized(best_c, r_target, margin=0.006)
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
                options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.08, 1.5e-4)
            else:
                step *= 0.4
                if step < 5e-8:
                    break
        except Exception:
            break
    return best_c, best_r


def _ultra_polish(c0, r0, max_time=10.0):
    """Ultra-precise final refinement using very low temperature focused energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    temps = [0.0003, 0.00015, 0.00008, 0.00004, 0.00002, 0.00001]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 0.5:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 150}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _randomize_clusters(c0, magnitude):
    """Randomize some clusters to create diversity."""
    n = c0.shape[0]
    r_est = _compute_radius(c0)
    tree = cKDTree(c0)
    dist_limit = 2.5 * r_est
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
        neighbors = tree.query_ball_point(c0[i], dist_limit)
        for j in neighbors:
            if i < j:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    cluster_list = list(clusters.values())
    new_c = c0.copy()
    n_randomize = max(1, len(cluster_list) // 4)
    random_clusters = np.random.choice(len(cluster_list), n_randomize, replace=False)
    for ci in random_clusters:
        cluster = cluster_list[ci]
        angle = np.random.random() * 2 * np.pi
        mag = np.random.exponential(magnitude)
        dx = mag * np.cos(angle)
        dy = mag * np.sin(angle)
        for idx in cluster:
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
    return np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)


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

    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(35.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Focused energy optimization
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(8.0, deadline - time.time() - 40)
        c_foc, r_foc = _local_optimize_focused(best_c, max_time=min(18.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()

    # 4. Rigid-body dynamics relaxation (MD)
    if best_c is not None and time.time() < deadline - 30:
        remaining = max(10.0, deadline - time.time() - 30)
        c_md, r_md = _md_relaxation(best_c, best_radius, max_time=min(25.0, remaining))
        if r_md > best_radius:
            best_radius = r_md
            best_c = c_md.copy()

    # 5. Another smooth optimization pass
    if best_c is not None and time.time() < deadline - 22:
        remaining = max(5.0, deadline - time.time() - 22)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(12.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()

    # 6. Focused energy refinement after MD
    if best_c is not None and time.time() < deadline - 14:
        remaining = max(5.0, deadline - time.time() - 14)
        c_foc2, r_foc2 = _local_optimize_focused(best_c, max_time=min(8.0, remaining))
        if r_foc2 > best_radius:
            best_radius = r_foc2
            best_c = c_foc2.copy()

    # 7. Final high-precision SLSQP polish
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(10.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    # 8. Ultra-precise final refinement
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(2.0, deadline - time.time() - 0.5)
        c_ultra, r_ultra = _ultra_polish(best_c, best_radius, max_time=min(5.0, remaining))
        if r_ultra > best_radius:
            best_radius = r_ultra
            best_c = c_ultra

    if best_c is None:
        best_c = _initialize_hexagonal()
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
