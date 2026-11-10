"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _fibonacci_lattice():
    """Create a Fibonacci lattice (sunflower pattern) for uniform distribution."""
    golden_ratio = (1 + np.sqrt(5)) / 2.0
    angles = 2 * np.pi * np.arange(N) / golden_ratio
    # Radius scales with sqrt to fill the square uniformly
    radii = np.sqrt(np.arange(N) / N) * 0.5
    centres = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    return centres


def _energy_and_gradient(centres_flat, r_target, softness=0.01):
    """
    Compute energy and gradient for a soft repulsive potential.
    Energy = sum of soft repulsion terms + wall penalties.
    Uses log-sum-exp style soft-min for smoothness.
    """
    centres = centres_flat.reshape(N, 2)
    
    # Wall penalty: push centres away from walls
    wall_energy = 0.0
    wall_grad = np.zeros_like(centres)
    for axis in range(2):
        # Left wall
        viol_l = np.maximum(0, (-0.5 - centres[:, axis]) + softness)
        wall_energy += np.sum(viol_l ** 2)
        wall_grad[:, axis] += 2 * viol_l * (-1)
        # Right wall
        viol_r = np.maximum(0, (centres[:, axis] - 0.5) + softness)
        wall_energy += np.sum(viol_r ** 2)
        wall_grad[:, axis] += 2 * viol_r * (1)
    
    # Pairwise repulsion: soft potential U = sum_{i<j} exp(-dist_ij / r_target)
    # This creates a smooth energy landscape that penalizes close pairs
    energy = wall_energy
    grad = wall_grad.copy()
    
    # Compute pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    
    # Only consider pairs closer than 5*r_target for efficiency
    mask = dists < 5.0 * r_target
    np.fill_diagonal(mask, False)
    
    for i in range(N):
        close_idx = np.where(mask[i])[0]
        if len(close_idx) == 0:
            continue
        
        d = dists[i, close_idx]
        d = np.maximum(d, 1e-10)
        
        # Soft repulsion: U = exp(-d / r_target)
        # dU/dd = -exp(-d/r_target) / r_target
        u = np.exp(-d / r_target)
        du_dd = -u / r_target
        
        # Gradient contribution: dU/dx_i = du_dd * d(d)/d(x_i)
        # d(d)/d(x_i) = -(x_j - x_i) / d
        deltas = centres[close_idx] - centres[i]
        grad_contrib = np.zeros(2)
        for k, j in enumerate(close_idx):
            grad_contrib += du_dd[k] * (-deltas[k] / d[k])
        
        grad[i] += grad_contrib
        energy += np.sum(u)
    
    return energy, grad.flatten()


def _lbfgs_optimize(centres, r_target, max_time=30.0):
    """
    Optimize positions using L-BFGS-B to minimize soft repulsion energy.
    The target radius r_target defines the scale of the repulsive potential.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    def objective_and_grad(x):
        if time.time() > deadline:
            return 0.0, np.zeros_like(x)
        e, g = _energy_and_gradient(x, r_target)
        return e, g
    
    x0 = centres.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        new_centres = result.x.reshape(N, 2)
        # Clip to valid range
        new_centres = np.clip(new_centres, -0.5 + 1e-9, 0.5 - 1e-9)
        
        return new_centres
    
    except Exception:
        return centres


def _binary_search_radius(centres, r_low, r_high, max_time=60.0):
    """
    Binary search for the maximum achievable radius.
    For each candidate radius, run L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Initial bracket
    if best_r > r_high:
        r_low = r_high
        r_high = best_r
    
    iterations = 0
    while r_high - r_low > 1e-7 and time.time() < deadline:
        iterations += 1
        r_mid = (r_low + r_high) / 2
        
        # Try to optimize with this target radius
        optimized = _lbfgs_optimize(centres, r_mid, max_time=min(20.0, deadline - time.time()))
        
        # Check if the optimized configuration achieves at least r_mid
        actual_r = _compute_radius(optimized)
        
        if actual_r >= r_mid - 1e-9:
            # We can achieve this radius, try higher
            r_low = r_mid
            best_centres = optimized
            best_r = actual_r
        else:
            # Cannot achieve this radius, try lower
            r_high = r_mid
    
    return best_centres, best_r


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 20):
        # For hexagonal packing, alternating rows have offset
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_start = -0.5 + s
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _local_optimization(centres, max_iters=500, initial_step=0.01):
    """Local optimization using gradient-free method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    step_size = initial_step
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate constraints
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)
        
        # Try perturbations on the most constrained point
        improved = False
        n_tries = 30
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            delta = step_size * np.array([np.cos(angle), np.sin(angle)])
            new_centres = best_centres.copy()
            new_centres[idx] += delta
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                step_size *= 0.95
                break
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return best_centres, best_r


def _simulated_annealing(centres, max_iters=1000, t_start=0.01, t_end=1e-6):
    """Simulated annealing optimization."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_centres = centres.copy()
    current_r = best_r
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, t)
        delta = magnitude * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        # Check bounds
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
        else:
            # Accept with probability exp((new_r - current_r) / t)
            delta_r = new_r - current_r
            if np.random.random() < np.exp(delta_r / max(t, 1e-12)):
                current_centres = new_centres
                current_r = new_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Try multiple initialization strategies and keep the best
    candidates = []
    
    # 1. Fibonacci lattice
    fib_centres = _fibonacci_lattice()
    fib_r = _compute_radius(fib_centres)
    candidates.append((fib_centres, fib_r))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    hex_r = _compute_radius(hex_centres)
    candidates.append((hex_centres, hex_r))
    
    # 3. Grid packing
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    grid_r = _compute_radius(grid)
    candidates.append((grid, grid_r))
    
    # Pick the best initial configuration
    candidates.sort(key=lambda x: x[1], reverse=True)
    centres, current_r = candidates[0]
    
    # L-BFGS-B optimization with binary search on radius
    if time.time() - start_time < TIME_LIMIT - 30:
        # First, do a quick L-BFGS optimization to improve the initial config
        r_estimate = current_r
        optimized = _lbfgs_optimize(centres, r_estimate, max_time=20.0)
        opt_r = _compute_radius(optimized)
        if opt_r > current_r:
            centres = optimized
            current_r = opt_r
        
        # Binary search for better radius
        if time.time() - start_time < TIME_LIMIT - 15:
            r_low = current_r * 0.95
            r_high = current_r * 1.15
            centres, current_r = _binary_search_radius(centres, r_low, r_high, max_time=TIME_LIMIT - 15 - (time.time() - start_time))
    
    # Final local optimization polish
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _local_optimization(centres, max_iters=100, initial_step=0.005)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
