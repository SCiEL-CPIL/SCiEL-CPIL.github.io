"""Pack 50 equal circles in a unit square [-1/2, 1/2]^2 to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
SIDE = 1.0
HALF = 0.5

# Known good packing radius from Boll, Donovan, Graham, Lubachevsky (2000)
# r_best = 0.071377103865
# We aim to beat the simple grid (score 1.0) which gives r = 1/(2*sqrt(50)) ≈ 0.07071

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], HALF + centers[:, 0]),
        np.minimum(HALF - centers[:, 1], HALF + centers[:, 1])
    )
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _objective_and_grad(flat):
    """
    Objective: minimize -(radius) to maximize radius.
    Use a smooth approximation: minimize the sum of penalties for violations
    plus a term that pushes all distances up.
    
    Actually, let's use a different approach: 
    We want to maximize r such that all pairwise distances >= 2r and all wall distances >= r.
    This is equivalent to minimizing the maximum violation.
    
    Better approach: use a smooth min-like function.
    We minimize: max(0, 2r - d_ij) for all pairs + max(0, r - wall_i) for all walls.
    But we don't have r as a variable. 
    
    Alternative: directly minimize -min(min_pair/2, min_wall) but that's non-smooth.
    
    Let's use: minimize the sum of soft-min of all constraints.
    """
    centers = flat.reshape(-1, 2)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], HALF + centers[:, 0]),
        np.minimum(HALF - centers[:, 1], HALF + centers[:, 1])
    )
    
    # All constraints: pairwise/2 and wall distances
    # We want to maximize the minimum of all these.
    # Use a smooth approximation: -log-sum-exp(-constraints) approximates min(constraints)
    # But that's expensive. Instead, let's just return the negative of the min.
    
    all_constraints = np.concatenate([dists / 2.0, wall_dists])
    min_val = np.min(all_constraints)
    
    # Gradient: we need d(min)/d(centers)
    # This is tricky. Let's use a different formulation.
    
    return -min_val


def _construct_hexagonal():
    """Construct a hexagonal packing as initial guess."""
    # Hexagonal packing in a square
    # For N=50, we can try various arrangements
    # Let's try a 7x7 grid with some removed, or other arrangements
    
    # Simple approach: start with a grid and use hexagonal perturbation
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid
    xs = np.linspace(-HALF + 0.5/m, HALF - 0.5/m, m)
    ys = np.linspace(-HALF + 0.5/m, HALF - 0.5/m, m)
    grid = np.array([(x, y) for y in ys for x in xs])
    
    # Try hexagonal offset
    hex_grid = []
    row = 0
    for y in ys:
        for x in xs:
            offset = 0.5 / m if row % 2 == 0 else 0.0
            hex_grid.append((x + offset, y))
        row += 1
    
    hex_grid = np.array(hex_grid[:N])
    return hex_grid


def _local_optimize(centers, time_budget=10.0):
    """
    Local optimization using L-BFGS-B to maximize the minimum distance.
    """
    start_time = time.time()
    
    def neg_radius(flat):
        centers_local = flat.reshape(-1, 2)
        r = _compute_radius(centers_local)
        return -r
    
    def neg_radius_grad(flat):
        centers_local = flat.reshape(-1, 2)
        n = centers_local.shape[0]
        
        # Compute all pairwise distances
        diff = centers_local[:, np.newaxis, :] - centers_local[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(HALF - centers_local[:, 0], HALF + centers_local[:, 0]),
            np.minimum(HALF - centers_local[:, 1], HALF + centers_local[:, 1])
        )
        
        all_constraints = np.concatenate([dists / 2.0, wall_dists])
        min_val = np.min(all_constraints)
        
        # Gradient: subgradient of min
        grad_flat = np.zeros_like(flat)
        
        # Find which constraints are at the minimum
        min_idx = np.argmin(all_constraints)
        
        if min_idx < len(dists):
            # It's a pairwise constraint
            # Get the pair
            i, j = np.unravel_index(min_idx, dists.shape)
            if i == j:
                return 0.0, np.zeros_like(flat)
            # d/dx_i of (|x_i - x_j|/2) = (x_i - x_j) / (2 * |x_i - x_j|)
            # d/dx_j of (|x_i - x_j|/2) = -(x_i - x_j) / (2 * |x_i - x_j|)
            d = dists[i, j]
            if d > 1e-15:
                vec = (centers_local[i] - centers_local[j]) / (2 * d)
                grad_flat[2*i:2*i+2] += vec
                grad_flat[2*j:2*j+2] -= vec
        else:
            # It's a wall constraint
            wall_idx = min_idx - len(dists)
            # Wall distance for point wall_idx
            x, y = centers_local[wall_idx]
            # d/dx of min(HALF-x, HALF+x, HALF-y, HALF+y)
            # The min is achieved by one of the four
            wall_vals = np.array([HALF - x, HALF + x, HALF - y, HALF + y])
            min_wall_idx = np.argmin(wall_vals)
            if min_wall_idx == 0:
                grad_flat[2*wall_idx] -= 1.0
            elif min_wall_idx == 1:
                grad_flat[2*wall_idx] += 1.0
            elif min_wall_idx == 2:
                grad_flat[2*wall_idx+1] -= 1.0
            elif min_wall_idx == 3:
                grad_flat[2*wall_idx+1] += 1.0
        
        return -min_val, -grad_flat
    
    # Use L-BFGS-B with bounds
    bounds = [(-HALF, HALF)] * (2 * N)
    flat_init = centers.flatten()
    
    best_flat = flat_init.copy()
    best_val = neg_radius(flat_init)
    
    while time.time() - start_time < time_budget:
        result = minimize(
            neg_radius,
            best_flat,
            jac=neg_radius_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_flat = result.x.copy()
        
        # Check if we've converged
        if result.success and result.fun >= best_val - 1e-10:
            break
        
        # Add small perturbation to escape local minima
        perturbation = 0.01 * np.random.randn(len(best_flat))
        best_flat = best_flat + perturbation
        best_flat = np.clip(best_flat, -HALF, HALF)
    
    return best_flat.reshape(-1, 2)


def construct():
    """Main function: pack 50 circles in unit square to maximize radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with hexagonal-like initial guess
    centers = _construct_hexagonal()
    
    # Also try a few different initial configurations
    initial_configs = [centers]
    
    # Add a standard grid
    m = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(-HALF + 0.5/m, HALF - 0.5/m, m)
    ys = np.linspace(-HALF + 0.5/m, HALF - 0.5/m, m)
    grid = np.array([(x, y) for y in ys for x in xs])[:N]
    initial_configs.append(grid)
    
    # Add random configurations
    np.random.seed(42)
    for _ in range(3):
        rand_centers = np.random.uniform(-HALF + 0.05, HALF - 0.05, (N, 2))
        initial_configs.append(rand_centers)
    
    best_centers = initial_configs[0]
    best_radius = _compute_radius(best_centers)
    
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break
        
        time_budget = min(20.0, deadline - time.time() - 5.0)
        if time_budget < 5.0:
            break
        
        optimized = _local_optimize(init, time_budget=time_budget)
        radius = _compute_radius(optimized)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized
    
    # Final refinement on best
    if time.time() < deadline - 10.0:
        time_budget = min(15.0, deadline - time.time() - 5.0)
        if time_budget > 5.0:
            final = _local_optimize(best_centers, time_budget=time_budget)
            final_radius = _compute_radius(final)
            if final_radius > best_radius:
                best_centers = final
                best_radius = final_radius
    
    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    
    return best_centers
