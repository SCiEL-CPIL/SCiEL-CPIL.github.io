"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_objective(params, beta=50.0):
    """
    Smooth surrogate for the negative radius using soft-min (log-sum-exp).
    We want to maximize min(pairwise_dists/2, boundary_dists).
    Soft-min approximates the minimum: -soft_min = soft_max of negatives.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    # Soft-min: -log-sum-exp(-beta * x) / beta approximates min(x) for large beta
    # We want to maximize the soft-min, so minimize negative soft-min
    # Numerically stable computation
    min_val = np.min(all_vals)
    shifted = beta * (all_vals - min_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -700, 700)
    lse = np.log(np.sum(np.exp(shifted))) - beta * min_val
    soft_min_val = min_val + lse / beta
    
    return -soft_min_val


def _active_set_polish(c0, max_time=15.0, tol=1e-12):
    """
    High-precision polish using only active (tight) constraints.
    After smooth optimization, identify pairs and boundary distances
    that are within a small margin of the minimum radius, then optimize
    just those constraints with tight tolerances.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    n = c.shape[0]
    
    # Identify active constraints
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    boundary_dists = 0.5 - np.abs(c)
    
    # Active pairs: those within 1.05*r of the min radius
    margin = 0.05 * r + 1e-6
    active_mask = pair_vals < (r + margin)
    active_pairs = np.where(active_mask)[0]
    
    # Active boundary constraints
    active_boundary = np.where(boundary_dists.flatten() < (r + margin))[0]
    
    # Build index mapping for active pairs
    # pdist uses condensed indexing: pair (i, j) with i < j has index
    # i*n - i*(i+1)//2 + j - i - 1
    def _pair_index(i, j):
        return i * n - i * (i + 1) // 2 + j - i - 1
    
    active_i = []
    active_j = []
    for idx in active_pairs:
        # Invert condensed index
        # Find i such that i*(2n-1-i)/2 <= idx < (i+1)*(2n-1-i-1)/2
        i = 0
        while i < n - 1:
            count = n - i - 1
            if idx < i * count + count:
                break
            idx -= count
            i += 1
        j = i + 1 + idx
        active_i.append(i)
        active_j.append(j)
    
    active_i = np.array(active_i)
    active_j = np.array(active_j)
    active_boundary = np.array(active_boundary)
    
    if len(active_i) == 0 and len(active_boundary) == 0:
        return c, r
    
    def _active_obj(params):
        c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Compute active pair distances
        if len(active_i) > 0:
            diffs = c_local[active_i] - c_local[active_j]
            pair_dists = np.sqrt(np.sum(diffs**2, axis=1)) / 2.0
        else:
            pair_dists = np.array([1.0])
        
        # Compute active boundary distances
        if len(active_boundary) > 0:
            bi = active_boundary // 2
            bj = active_boundary % 2
            bd = np.where(bj == 0, 0.5 - c_local[bi, 0], 0.5 - c_local[bi, 1])
            bd = np.where(bj == 1, 0.5 + c_local[bi, 1], bd)
        else:
            bd = np.array([1.0])
        
        all_vals = np.concatenate([pair_dists, bd])
        return -np.min(all_vals)
    
    try:
        result = minimize(
            _active_obj,
            c.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > r:
            c = c_new
            r = r_new
    except Exception:
        pass
    
    # Second pass with even tighter active set
    if time.time() < start_time + max_time - 2:
        r = _compute_radius(c)
        dists = pdist(c, metric='euclidean')
        pair_vals = dists / 2.0
        boundary_dists = 0.5 - np.abs(c)
        
        margin = 0.02 * r + 1e-7
        active_mask = pair_vals < (r + margin)
        active_pairs = np.where(active_mask)[0]
        active_boundary = np.where(boundary_dists.flatten() < (r + margin))[0]
        
        active_i = []
        active_j = []
        for idx in active_pairs:
            i = 0
            while i < n - 1:
                count = n - i - 1
                if idx < i * count + count:
                    break
                idx -= count
                i += 1
            j = i + 1 + idx
            active_i.append(i)
            active_j.append(j)
        
        active_i = np.array(active_i)
        active_j = np.array(active_j)
        active_boundary = np.array(active_boundary)
        
        if len(active_i) > 0 or len(active_boundary) > 0:
            def _active_obj2(params):
                c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                
                if len(active_i) > 0:
                    diffs = c_local[active_i] - c_local[active_j]
                    pair_dists = np.sqrt(np.sum(diffs**2, axis=1)) / 2.0
                else:
                    pair_dists = np.array([1.0])
                
                if len(active_boundary) > 0:
                    bi = active_boundary // 2
                    bj = active_boundary % 2
                    bd = np.where(bj == 0, 0.5 - c_local[bi, 0], 0.5 - c_local[bi, 1])
                    bd = np.where(bj == 1, 0.5 + c_local[bi, 1], bd)
                else:
                    bd = np.array([1.0])
                
                all_vals = np.concatenate([pair_dists, bd])
                return -np.min(all_vals)
            
            try:
                result = minimize(
                    _active_obj2,
                    c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100}
                )
                
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > r:
                    c = c_new
                    r = r_new
            except Exception:
                pass
    
    return c, r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
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


def _force_directed_init(c0, n_iter=500, k=0.005):
    """
    Force-directed relaxation: circles repel each other and are attracted to center.
    This helps untangle overlaps from random initialization.
    """
    c = c0.copy()
    n = c.shape[0]
    
    for it in range(n_iter):
        # Compute repulsive forces between all pairs
        # For efficiency, only consider pairs closer than some threshold
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion: F = k / dist^2 * direction (capped to avoid blowup)
        min_dist = 0.01
        safe_dists = np.maximum(dists, min_dist)
        forces = np.zeros_like(c)
        
        # Only compute for close pairs to save time
        mask = safe_dists < 0.15
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = safe_dists[i, j]
                    f = k / (d * d)
                    forces[i] += f * (c[i] - c[j]) / d
                    forces[j] -= f * (c[i] - c[j]) / d
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Damping
        damping = 0.9 * (1.0 - it / n_iter) + 0.1
        c += damping * forces
        
        # Clip to bounds
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c


def _local_optimize_smooth(c0, max_time=25.0, beta_start=20.0, beta_end=200.0):
    """
    Local optimization using L-BFGS-B with a smooth surrogate objective.
    Uses annealing of beta to gradually sharpen the soft-min toward the true min,
    followed by an active-set polish for high precision.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Smooth optimization with increasing beta
    betas = [beta_start, 40.0, 80.0, 150.0, beta_end]
    
    for beta in betas:
        if time.time() > start_time + max_time - 8:
            break
        
        def _obj(params):
            return _soft_min_objective(params, beta=beta)
        
        try:
            result = minimize(
                _obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 2: Active-set polish for high precision
    if time.time() < start_time + max_time - 3:
        remaining = max_time - (time.time() - start_time) - 3
        if remaining > 2:
            c_polish, r_polish = _active_set_polish(best_c, max_time=remaining)
            if r_polish > best_r:
                best_c = c_polish
                best_r = r_polish
    
    return best_c, best_r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with smooth surrogate."""
    return _local_optimize_smooth(c0, max_time=max_time)


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Optimize hexagonal initialization with generous time
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(50.0, deadline - time.time() - 25))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try force-directed initialization from random starts
    np.random.seed(123)
    for trial in range(3):
        if time.time() > deadline - 35:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Apply force-directed relaxation
        c_fd = _force_directed_init(c_rand, n_iter=200, k=0.005)
        r_fd = _compute_radius(c_fd)
        
        if r_fd > best_radius:
            best_radius = r_fd
            best_c = c_fd.copy()
        
        if time.time() < deadline - 25:
            c_opt, r_opt = _local_optimize_smooth(c_fd, max_time=min(30.0, deadline - time.time() - 15))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final active-set polish on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            c_polish, r_polish = _active_set_polish(best_c, max_time=remaining)
            if r_polish > best_radius:
                best_radius = r_polish
                best_c = c_polish.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
