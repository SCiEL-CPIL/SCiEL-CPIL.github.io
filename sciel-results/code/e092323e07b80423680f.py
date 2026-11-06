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


def _get_closest_pairs(centers, k=15):
    """Get the k closest pairs of spheres."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    flat_idx = np.argsort(dist.flatten())
    pairs = []
    for idx in flat_idx[:k]:
        i = int(idx) // N
        j = int(idx) % N
        if i != j:
            pairs.append((i, j, dist[i, j]))
    return pairs


def _push_apart(centers, max_iters=300, step_scale=1.0):
    """Iteratively push apart the closest pairs to increase minimum distance."""
    centers = centers.copy()
    for it in range(max_iters):
        # Get the closest pairs
        pairs = _get_closest_pairs(centers, k=10)
        if not pairs:
            break
        
        # Find the minimum distance
        min_dist = min(p[2] for p in pairs)
        
        # Check wall constraints
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        # Determine effective radius
        eff_radius = min(min_dist / 2.0, min_wall)
        
        # If wall is the constraint, push inward
        if min_wall < min_dist / 2.0 - 1e-12:
            # Push spheres away from walls
            for i in range(N):
                for d in range(3):
                    if centers[i, d] > 0 and wall[i, d] <= min_wall + 1e-12:
                        centers[i, d] -= 0.001 * step_scale
                    elif centers[i, d] < 0 and wall[i, d] <= min_wall + 1e-12:
                        centers[i, d] += 0.001 * step_scale
        else:
            # Push apart the closest pairs
            for i, j, d in pairs:
                if d < 0.25:  # Only push if close
                    diff = centers[i] - centers[j]
                    if np.linalg.norm(diff) < 1e-12:
                        # Random perturbation for coincident points
                        rng = np.random.RandomState(i * 1000 + j)
                        centers[i] += rng.randn(3) * 0.001 * step_scale
                        centers[j] -= rng.randn(3) * 0.001 * step_scale
                    else:
                        diff_norm = np.linalg.norm(diff)
                        direction = diff / diff_norm
                        push = 0.002 * step_scale
                        centers[i] += direction * push
                        centers[j] -= direction * push
        
        # Clip to stay inside cube
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _make_grid_initial(seed=42):
    """Create a grid-based initial configuration."""
    rng = np.random.RandomState(seed)
    # For 50 spheres, use a 4x4x4 grid (64 positions) and take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter to break symmetry
    centers += rng.uniform(-0.005, 0.005, size=centers.shape)
    # Clip to stay inside cube
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_bcc_initial(seed=42):
    """Create a body-centered cubic inspired initial configuration."""
    rng = np.random.RandomState(seed)
    # Use a denser grid with some body-center points
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add body-center points for some cells
    body_centers = []
    dx = ticks[1] - ticks[0]
    for z in ticks[:-1]:
        for y in ticks[:-1]:
            for x in ticks[:-1]:
                bc = [x + dx/2, y + dx/2, z + dx/2]
                body_centers.append(bc)
    
    all_points = np.vstack([grid, np.array(body_centers)])
    # Shuffle and take N
    idx = rng.permutation(len(all_points))[:N]
    centers = all_points[idx].copy()
    centers += rng.uniform(-0.003, 0.003, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random_initial(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _neg_radius(centers_flat):
    """Negative radius for minimization."""
    centers = centers_flat.reshape(N, 3)
    return -_radius_for_centers(centers)


def _local_optimize(centers, time_budget=20.0):
    """Local optimization using L-BFGS on the negative radius."""
    start_time = time.time()
    current = centers.copy()
    current_radius = _radius_for_centers(current)
    
    x0 = current.flatten()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Use L-BFGS with numerical gradient
    for attempt in range(5):
        if time.time() - start_time > time_budget - 5.0:
            break
        try:
            result = minimize(
                _neg_radius,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if result.fun < -current_radius + 1e-12:
                new_centers = result.x.reshape(N, 3)
                new_radius = _radius_for_centers(new_centers)
                if new_radius > current_radius:
                    current = new_centers
                    current_radius = new_radius
                    x0 = result.x
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return current, current_radius


def _polish(centers, time_budget=10.0):
    """Final high-precision polish pass with tighter tolerances."""
    start_time = time.time()
    current = centers.copy()
    current_radius = _radius_for_centers(current)
    
    # Multi-scale push-apart for fine-tuning
    for scale in [0.1, 0.05, 0.02, 0.01, 0.005]:
        if time.time() - start_time > time_budget * 0.7:
            break
        new_centers = _push_apart(current, max_iters=100, step_scale=scale)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-14:
            current = new_centers
            current_radius = new_radius
    
    # Final L-BFGS polish with very tight tolerances
    if time.time() - start_time < time_budget * 0.9:
        x0 = current.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)
        try:
            result = minimize(
                _neg_radius,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if result.fun < -current_radius + 1e-14:
                new_centers = result.x.reshape(N, 3)
                new_radius = _radius_for_centers(new_centers)
                if new_radius > current_radius:
                    current = new_centers
                    current_radius = new_radius
        except Exception:
            pass
    
    return current, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    # Try multiple initializations
    initializations = [
        ("grid", _make_grid_initial(42)),
        ("grid2", _make_grid_initial(123)),
        ("grid3", _make_grid_initial(456)),
        ("bcc", _make_bcc_initial(42)),
        ("bcc2", _make_bcc_initial(777)),
        ("rand1", _make_random_initial(1)),
        ("rand2", _make_random_initial(2)),
    ]

    for name, init_centers in initializations:
        if time.time() > deadline - 30.0:
            break

        current = init_centers.copy()
        current_radius = _radius_for_centers(current)
        
        # First, do push-apart to get a better starting point
        current = _push_apart(current, max_iters=300, step_scale=1.0)
        current_radius = _radius_for_centers(current)
        
        # Then local optimization
        current, current_radius = _local_optimize(current, time_budget=25.0)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current.copy()

    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final polish pass on the best configuration
    if time.time() < deadline - 5.0:
        best_centers, best_radius = _polish(best_centers, time_budget=15.0)

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers
# EVOLVE-BLOCK-END
