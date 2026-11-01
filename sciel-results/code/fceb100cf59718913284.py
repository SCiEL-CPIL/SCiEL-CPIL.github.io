import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _violation(centers, r):
    """
    Compute sum of squared violations for radius r.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We want dists >= 2r. Violation is max(0, 2r - dists)
    pair_violation = np.sum(np.maximum(0, 2*r - dists)**2)
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists)**2)
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # d/dx max(0, 2r - d)^2 = 2 * max(0, 2r - d) * d/dx(2r - d)
    # d/dx (2r - d) = - (x_i - x_j) / d
    # So term is -2 * max(0, 2r - d) * (x_i - x_j) / d
    
    active = dists < 2*r
    # Avoid division by zero
    active[dists < 1e-12] = False
    
    for i in range(n):
        for j in range(i+1, n):
            if active[i, j]:
                d = dists[i, j]
                factor = 2 * (2*r - d) / d
                grad[i] -= factor * (centers[i] - centers[j])
                grad[j] += factor * (centers[i] - centers[j])
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    for i in range(n):
        if wall_dists_x[i] < r:
            # d/dx (r - (0.5 - |x|))^2 = 2(r - w) * sign(x)
            grad[i, 0] += 2 * (r - wall_dists_x[i]) * np.sign(centers[i, 0])
        if wall_dists_y[i] < r:
            grad[i, 1] += 2 * (r - wall_dists_y[i]) * np.sign(centers[i, 1])
    
    return grad

def _optimize_for_radius(start_centers, target_r, max_iter=50):
    """
    Use L-BFGS-B to minimize violation for a target radius.
    Returns (centers, violation).
    """
    x0 = start_centers.flatten()
    
    def obj(x):
        c = x.reshape(-1, 2)
        return _violation(c, target_r)
    
    def grad(x):
        c = x.reshape(-1, 2)
        return _grad_violation(c, target_r).flatten()
        
    try:
        result = opt.minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return result.x.reshape(-1, 2), _violation(result.x.reshape(-1, 2), target_r)
    except Exception:
        return start_centers, _violation(start_centers, target_r)

def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    # 7x7 grid is 49 points. We need 50.
    # Let's use a 7x8 grid and pick best 50?
    # Or just 7x7 with one extra point in the middle of a cell?
    # Simpler: 7x7 grid, then add one point at (0,0) if not occupied?
    # Actually, standard grid:
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) < N:
        # Add more points
        # This is getting complicated. Let's just use a dense random grid or hex.
        pass
    
    # Let's use a hexagonal packing initialization
    # Hex grid spacing s.
    # Area per circle ~ s^2 * sqrt(3)/2.
    # Total area 1. N=50. s ~ sqrt(1 / (50 * 0.866)) ~ 0.15
    # Radius ~ s/2 ~ 0.075.
    
    points = []
    s = 0.15
    dx = s
    dy = s * np.sqrt(3) / 2
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            points.append([x, y])
            
    points = np.array(points)
    # Sort by distance to center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Scale and center
    # We want to fit in [-0.5, 0.5]
    # Current extent is approx 3*0.15 = 0.45. Fits.
    
    # Add jitter
    rng = np.random.default_rng(1)
    points = points + rng.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    # If we have more than N, take first N. If less, pad.
    if len(points) < N:
        # Pad with random points
        extra = rng.uniform(-0.5, 0.5, (N - len(points), 2))
        points = np.vstack([points, extra])
    else:
        points = points[:N]
        
    return points

def _random_init(rng):
    return rng.uniform(-0.5, 0.5, (N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations
    inits = []
    # 1. Hex
    inits.append(_grid_init())
    # 2. Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # We have 49. Add one at (0,0) is bad.
    # Let's just take first 50 from a larger grid?
    # 8x8 grid = 64 points.
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Sort by distance to center to pick best 50?
    dists = np.sum(grid**2, axis=1)
    idx = np.argsort(dists)
    grid = grid[idx][:N]
    rng = np.random.default_rng(2)
    grid = grid + rng.normal(0, 0.01, grid.shape)
    inits.append(grid)
    
    # 3. Random
    rng = np.random.default_rng(3)
    inits.append(_random_init(rng))
    
    best_centers = None
    best_radius = 0.0
    
    # Initial radii
    for init in inits:
        r = _compute_radius(init)
        if best_radius is None or r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    # Binary search for better radius
    lo = best_radius
    hi = 0.08 # Upper bound
    
    # We will iterate a few times, trying to improve
    # To be safe, we limit the number of L-BFGS-B calls
    
    # Strategy: Try to improve by small steps
    # Or binary search.
    
    # Let's do a few iterations of "try to find feasible for r_mid"
    
    # To save time, we only run optimization if we have time.
    
    # We'll try 10 binary search steps.
    for _ in range(10):
        if time.time() > deadline - 10:
            break
            
        mid = (lo + hi) / 2
        # Try to find a configuration for radius mid
        # Use the best current centers as start
        # Also try other inits if time permits? No, stick to best.
        
        trial_centers, violation = _optimize_for_radius(best_centers, mid, max_iter=50)
        
        trial_r = _compute_radius(trial_centers)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best_centers = trial_centers
            best_radius = trial_r
        else:
            hi = mid
            
    # Final polish: Try to maximize radius directly from best_centers
    # Use a smooth objective
    if time.time() < deadline - 5:
        def soft_min_obj(x, tau=0.005):
            centers = x.reshape(-1, 2)
            wall_dists_x = 0.5 - np.abs(centers[:, 0])
            wall_dists_y = 0.5 - np.abs(centers[:, 1])
            walls = np.minimum(wall_dists_x, wall_dists_y)
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_dists = dists / 2.0
            
            all_dists = np.concatenate([walls, pair_dists[np.triu_indices(N, k=1)]])
            
            min_d = np.min(all_dists)
            shifted = all_dists - min_d
            log_sum = np.log(np.sum(np.exp(-shifted / tau)))
            soft_min = min_d - tau * log_sum
            return -soft_min
            
        def soft_min_grad(x, tau=0.005):
            # Gradient of soft-min is complex. Let's use numerical or skip grad.
            # L-BFGS-B can work without grad, but it's slower.
            # Given the time constraint, we might skip this if it's too slow.
            # Or use finite differences.
            pass
            
        # Let's just do a few more iterations of violation minimization with increasing r
        # starting from best_radius
        current_r = best_radius
        for i in range(5):
            if time.time() > deadline - 2:
                break
            target_r = current_r * 1.001
            trial_centers, violation = _optimize_for_radius(best_centers, target_r, max_iter=20)
            trial_r = _compute_radius(trial_centers)
            if trial_r > current_r:
                current_r = trial_r
                best_centers = trial_centers
                best_radius = trial_r
            else:
                break
                
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return _grid_init()
        
    return best_centers
# EVOLVE-BLOCK-END
