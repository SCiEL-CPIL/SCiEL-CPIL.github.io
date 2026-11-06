"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_for_centers(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min over each coordinate of (0.5 - |c|)
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)


def _soft_radius(centers, alpha=50.0):
    """
    Smooth surrogate for the radius using log-sum-exp (soft-min).
    Returns a smooth approximation of min(min_pair/2, min_wall).
    """
    centers = np.asarray(centers)
    # Wall distances
    wall = 0.5 - np.abs(centers)
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    half_dist = 0.5 * np.sqrt(dist_sq + 1e-30)
    
    # Combine all constraints: wall distances and half-pairwise distances
    all_constraints = np.concatenate([wall.flatten(), half_dist[np.triu_indices(N, k=1)]])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-alpha * x))) / alpha
    # Clip to avoid overflow
    x = np.clip(all_constraints, 1e-10, 1.0)
    neg_alpha_x = -alpha * x
    max_val = np.max(neg_alpha_x)
    log_sum = max_val + np.log(np.sum(np.exp(neg_alpha_x - max_val)))
    soft_min = -log_sum / alpha
    
    return soft_min


def _force_directed(centers, iterations=200, target_r=None, time_limit=None):
    """
    Force-directed relaxation: all close pairs repel simultaneously.
    Much faster than single-pair push because it handles all interactions at once.
    """
    centers = centers.copy()
    start = time.time()
    
    # Determine target radius for repulsion
    if target_r is None:
        target_r = _radius_for_centers(centers) + 0.005
    
    for it in range(iterations):
        if time_limit is not None and time.time() - start > time_limit:
            break
        
        # Compute all pairwise forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N,N,3)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero
        
        # Repulsion force: strong when dist < 2*target_r
        # F = k * (target_r - dist/2) * direction/dist  when dist < 2*target_r
        force = np.zeros_like(centers)
        
        # Only consider pairs closer than 2*target_r
        mask = dist < 2.0 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            # For each close pair, add repulsive force
            # dist > 0 guaranteed by diagonal fill
            strength = np.where(mask, (target_r - dist / 2.0) / (dist + 1e-15), 0.0)
            # strength[i,j] is the magnitude of repulsion from j on i
            # Force on i from j: strength[i,j] * (centers[i] - centers[j]) / dist[i,j]
            # Sum over j
            # This is equivalent to: for each i, sum_j strength[i,j] * diff[i,j]
            # where diff[i,j] = centers[i] - centers[j]
            repulsive = strength[:, :, np.newaxis] * diff
            force = np.sum(repulsive, axis=1)
        
        # Wall repulsion: push spheres away from walls
        wall_dist = 0.5 - np.abs(centers)  # (N,3)
        wall_mask = wall_dist < target_r
        wall_force = np.zeros_like(centers)
        if np.any(wall_mask):
            # Push away from wall: direction is sign of center
            sign = np.sign(centers)
            # Where center is near 0, use random direction
            near_zero = np.abs(centers) < 1e-10
            sign[near_zero] = np.where(near_zero, 1.0, sign[near_zero])
            wall_strength = np.where(wall_mask, (target_r - wall_dist) / (target_r + 1e-15), 0.0)
            wall_force = wall_strength * sign
        
        # Apply forces with adaptive step size
        step = 0.01 * (1.0 - it / float(iterations + 1))
        centers += (force + wall_force) * step
        
        # Clip to stay in cube
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _neg_soft_radius(centers_flat, alpha=50.0):
    """Negative soft radius for smooth optimization."""
    centers = centers_flat.reshape(N, 3)
    r = _soft_radius(centers, alpha)
    return -r


def _make_grid_initial(seed, jitter=0.005):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += rng.uniform(-jitter, jitter, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_bcc_initial():
    """Create a BCC-like initialization for better starting point."""
    # 4x4x4 grid with BCC-like offsets
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    centers = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                centers.append([x, y, z])
                # Add BCC offset
                if len(centers) < N:
                    centers.append([x + 0.125, y + 0.125, z + 0.125])
    centers = np.array(centers[:N])
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _optimize_local(centers, time_limit=15.0):
    """Optimize a configuration using force-directed and smooth local refinement."""
    start = time.time()
    current = centers.copy()
    current_radius = _radius_for_centers(current)
    
    # Phase 1: Force-directed relaxation with decreasing target radius
    for scale in [1.0, 0.7, 0.4, 0.2]:
        if time.time() - start > time_limit * 0.4:
            break
        target_r = current_radius + 0.003 * scale
        new_centers = _force_directed(current, iterations=150, target_r=target_r,
                                       time_limit=time_limit * 0.1)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-12:
            current = new_centers
            current_radius = new_radius
    
    # Phase 2: Smooth optimization using soft-min surrogate
    if time.time() - start < time_limit * 0.75:
        x0 = current.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)
        
        # Try multiple alpha values for the soft-min
        for alpha in [30.0, 50.0, 80.0]:
            if time.time() - start > time_limit * 0.75:
                break
            try:
                result = minimize(
                    lambda x: _neg_soft_radius(x, alpha),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 300}
                )
                new_centers = result.x.reshape(N, 3)
                new_radius = _radius_for_centers(new_centers)
                if new_radius > current_radius + 1e-12:
                    current = new_centers
                    current_radius = new_radius
                    x0 = current.flatten()
            except Exception:
                pass
    
    # Phase 3: Final force-directed polish with tight target
    if time.time() - start < time_limit * 0.95:
        target_r = current_radius + 0.0005
        new_centers = _force_directed(current, iterations=100, target_r=target_r,
                                       time_limit=time_limit * 0.05)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-12:
            current = new_centers
            current_radius = new_radius
    
    return current, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    # Generate diverse initializations (fewer but better)
    initializations = []
    
    # Grid-based with small jitter
    for seed in [42, 123, 456]:
        initializations.append(_make_grid_initial(seed, jitter=0.003))
    
    # BCC-like initialization
    initializations.append(_make_bcc_initial())
    
    # One random initialization
    rng = np.random.RandomState(777)
    initializations.append(rng.uniform(-0.45, 0.45, size=(N, 3)))

    # Optimize each initialization
    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Allocate more time per initialization
        remaining = deadline - time.time()
        n_remaining = len(initializations) - init_idx
        time_per_init = min(20.0, remaining / (n_remaining + 1))
        
        if time_per_init < 5.0:
            break
        
        try:
            optimized, radius = _optimize_local(init_centers, time_limit=time_per_init)
        except Exception:
            continue
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()

    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers
# EVOLVE-BLOCK-END
