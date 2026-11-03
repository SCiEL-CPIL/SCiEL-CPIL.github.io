"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        # Edge case: single sphere
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return min_dist_to_wall
    
    # Compute pairwise distances
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to walls
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _repulsion_energy(centers, target_r):
    """Energy function: penalize distances less than 2*target_r between spheres
    and spheres too close to walls."""
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Repulsion: for pairs closer than 2*target_r, add penalty
    close_mask = dists < 2.0 * target_r
    if np.any(close_mask):
        # Sum of squared violations
        violations = (2.0 * target_r - dists[close_mask])
        pair_energy = np.sum(violations**2)
    else:
        pair_energy = 0.0
    
    # Wall repulsion: spheres should be at least target_r from walls
    wall_dists = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0, target_r - wall_dists)
    wall_energy = np.sum(wall_violations**2)
    
    return pair_energy + wall_energy

def _optimize_from_init(centers_init, target_r, max_iter=2000, tol=1e-10):
    """Optimize from a given initial configuration."""
    def objective(x):
        c = x.reshape(-1, 3)
        return _repulsion_energy(c, target_r)
    
    def gradient(x):
        c = x.reshape(-1, 3)
        # Compute gradient numerically or analytically
        # For simplicity, use finite differences (slow but safe)
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    result = minimize(
        objective,
        centers_init.flatten(),
        method='BFGS',
        jac=gradient,
        options={'maxiter': max_iter, 'gtol': tol, 'disp': False}
    )
    
    return result.x.reshape(-1, 3), result.success, result.fun

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    # Place points randomly in [-0.45, 0.45] to be safely inside
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Grid initialization
    strategies.append(('grid', _grid_init(N)))
    
    # Strategy 2: Random initializations
    for seed in range(20):
        strategies.append((f'random_{seed}', _random_init(N, seed=seed)))
    
    # First, estimate a good target radius from the grid
    for name, init in strategies:
        if time.time() > deadline - 5.0:
            break
        
        r_est = _compute_radius(init)
        if r_est > best_radius:
            best_radius = r_est
            best_centers = init.copy()
    
    # Now optimize from the best initializations
    # Sort strategies by initial radius (best first)
    scored_strategies = []
    for name, init in strategies:
        r_est = _compute_radius(init)
        scored_strategies.append((r_est, name, init))
    scored_strategies.sort(key=lambda x: -x[0])
    
    # Try to optimize from top initializations
    for r_est, name, init in scored_strategies[:10]:  # Top 10
        if time.time() > deadline - 10.0:
            break
        
        # Try to improve the radius
        # Start with current best radius as target
        target_r = max(best_radius, r_est)
        
        # Try optimization
        try:
            optimized, success, energy = _optimize_from_init(
                init, target_r, max_iter=1000, tol=1e-8
            )
            
            # Check if optimization improved the radius
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception as e:
            continue
    
    # Final polish: try a few more local optimizations from the best solution
    if best_centers is not None and time.time() < deadline - 20.0:
        # Try to improve further
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb slightly
            perturbed = best_centers + np.random.randn(N, 3) * 0.01
            # Clip to cube
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            target_r = best_radius
            try:
                optimized, success, energy = _optimize_from_init(
                    perturbed, target_r, max_iter=500, tol=1e-9
                )
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except:
                continue
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: ensure all centers are inside the cube
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
