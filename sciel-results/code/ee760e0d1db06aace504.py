"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the minimum radius for non-overlapping spheres inside [-0.5, 0.5]^3."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _initial_configuration_5x5x2():
    """Generate a 5x5x2 grid configuration (25 per layer)."""
    # 5x5 in xy plane, 2 layers in z
    xy_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = np.array([-0.25, 0.25])
    
    centers = []
    for z in z_ticks:
        for y in xy_ticks:
            for x in xy_ticks:
                centers.append([x, y, z])
    
    centers = np.array(centers)
    # We have 50 points exactly
    return centers


def _initial_configuration_grid():
    """Generate initial configuration using a cubic grid."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    return centers


def _local_refine(centers, max_iters=5000, base_step=0.01, rng_seed=123):
    """Local optimization: push overlapping spheres apart and keep in bounds.
    Uses a custom gradient-based approach on the max-min objective."""
    centers = centers.copy()
    rng = np.random.default_rng(rng_seed)
    n = len(centers)
    
    # Precompute indices for efficiency
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    step = base_step
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair (bottleneck)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[min_idx]
        
        # Compute wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Determine current bottleneck
        if min_dist / 2.0 < min_wall_dist:
            # Pairwise distance is bottleneck
            i, j = min_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = min_dist
            if dist < 1e-10:
                d = rng.normal(0, 0.001, 3)
                dist = np.sqrt(np.sum(d**2))
            direction = d / dist
            # Push apart
            centers[i] += step * direction
            centers[j] -= step * direction
        else:
            # Wall distance is bottleneck
            i, k = min_wall_idx
            if centers[i, k] > 0:
                centers[i, k] -= step
            else:
                centers[i, k] += step
        
        # Keep in bounds
        centers = np.clip(centers, -0.499, 0.499)
        
        # Compute radius
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Decay step size
        step *= 0.998
    
    return best_centers


def _global_refine(centers, max_iters=2000, rng_seed=456):
    """Global refinement: try random perturbations and keep improvements."""
    centers = centers.copy()
    rng = np.random.default_rng(rng_seed)
    n = len(centers)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        # Random perturbation
        idx = rng.integers(0, n)
        perturb = rng.normal(0, 0.005, 3)
        centers[idx] += perturb
        centers[idx] = np.clip(centers[idx], -0.499, 0.499)
        
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        else:
            centers[idx] -= perturb
    
    return best_centers


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with 5x5x2 grid which is a better structural starting point
    best_centers = _initial_configuration_5x5x2()
    best_radius = _compute_radius(best_centers)
    
    # Multiple refinement passes
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Try different initial configurations
        if restart % 3 == 0:
            centers = _initial_configuration_5x5x2()
            rng = np.random.default_rng(restart * 100)
            centers += rng.normal(0, 0.01, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        elif restart % 3 == 1:
            centers = _initial_configuration_grid()
            rng = np.random.default_rng(restart * 100 + 1)
            centers += rng.normal(0, 0.02, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        else:
            centers = _initial_configuration_5x5x2()
            rng = np.random.default_rng(restart * 100 + 2)
            # Swap some points
            idx = rng.choice(N, size=3, replace=False)
            centers[idx] = rng.uniform(-0.4, 0.4, size=(len(idx), 3))
            centers = np.clip(centers, -0.49, 0.49)
        
        # Local refinement
        centers = _local_refine(centers, max_iters=3000, base_step=0.008, rng_seed=restart)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius < 0.1:  # Sanity check
        best_centers = _initial_configuration_5x5x2()
    
    return best_centers
# EVOLVE-BLOCK-END
