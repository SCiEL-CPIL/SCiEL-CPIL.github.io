"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(x, r, beta=50.0):
    """
    Energy function for radius r: penalizes pairs closer than 2r and spheres outside cube.
    Returns (energy, gradient).
    """
    centers = x.reshape(N, 3)
    
    # Pairwise energy: sum of max(0, 2r - d_ij)^2 for all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Only consider pairs where 2r > d_ij, i.e., sq_dists < (2r)^2
    threshold = (2.0 * r) ** 2
    mask = sq_dists < threshold
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    if np.any(mask):
        # For violating pairs, compute penalty
        violating_dists = np.sqrt(np.maximum(sq_dists[mask], 0.0))
        violations = np.maximum(2.0 * r - violating_dists, 0.0)
        energy += np.sum(violations ** 2)
        
        # Gradient: d/d(c_i) sum_{j} max(0, 2r - |c_i - c_j|)^2
        # = 2 * sum_{j: violating} (2r - |c_i - c_j|) * (c_i - c_j) / |c_i - c_j|
        # Build the full matrix for gradient computation
        full_violations = np.zeros_like(sq_dists)
        full_violations[mask] = np.maximum(2.0 * r - np.sqrt(np.maximum(sq_dists[mask], 0.0)), 0.0)
        
        # Gradient contribution from pair i,j: 2 * (2r - d_ij) * (c_i - c_j) / d_ij
        # Sum over j for each i
        dist_matrix = np.sqrt(np.maximum(sq_dists, 1e-20))
        np.fill_diagonal(dist_matrix, 1.0)  # avoid div by zero on diagonal
        
        # grad_i = 2 * sum_j [violations[i,j] * (c_i - c_j) / d_ij]
        # where violations[i,j] = max(0, 2r - d_ij)
        weight_matrix = full_violations / dist_matrix
        weight_matrix[~mask] = 0.0
        
        # For each i: grad_i = 2 * sum_j weight[i,j] * (c_i - c_j)
        # = 2 * (c_i * sum_j weight[i,j] - sum_j weight[i,j] * c_j)
        sum_w = np.sum(weight_matrix, axis=1)  # (N,)
        weighted_sum = weight_matrix @ centers  # (N, 3)
        
        grad_pair = 2.0 * (sum_w[:, np.newaxis] * centers - weighted_sum)
        grad += grad_pair
    
    # Wall energy: sum of max(0, |c_ik| + r - 0.5)^2 for all i,k
    wall_violations = np.maximum(np.abs(centers) + r - 0.5, 0.0)
    energy += np.sum(wall_violations ** 2)
    
    # Gradient for wall: d/d(c_ik) max(0, |c_ik| + r - 0.5)^2
    # = 2 * max(0, |c_ik| + r - 0.5) * sign(c_ik)
    signs = np.sign(centers)
    signs[np.abs(centers) < 1e-15] = 1.0
    grad_wall = 2.0 * wall_violations * signs
    grad += grad_wall
    
    return float(energy), grad.flatten()

def _minimize_energy(init_centers, r, max_iter=2000, time_limit=10.0, beta=50.0):
    """Minimize energy for given radius r using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        return _energy_and_gradient(x, r, beta)
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'maxfun': 5000}
    )
    
    return result.x.reshape(N, 3)

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration to break symmetry."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _body_centered_init(n):
    """Generate a body-centered cubic inspired initialization."""
    grid = _grid_init(n)
    rng = np.random.RandomState(42)
    noise = rng.uniform(-0.005, 0.005, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            push = (r - dist/2.0) * d / dist * 0.5
            centers[i] += push
            centers[j] -= push
        else:
            i, k = min_wall_idx
            wall = wall_dists[i, k]
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _soft_min_objective_and_grad(x, temperature=0.001):
    """Soft-min radius objective for final polishing."""
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        exp_terms = np.zeros_like(sq_dists)
        exp_terms[finite_mask] = np.exp(-(sq_dists[finite_mask] - max_sq) / (4.0 * T2))
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq_pair = max_sq / 4.0 - T2 * log_sum
        else:
            soft_min_sq_pair = max_sq / 4.0
    else:
        soft_min_sq_pair = 0.0
    
    soft_min_sq_pair = max(soft_min_sq_pair, 0.0)
    soft_min_pair = np.sqrt(soft_min_sq_pair)
    
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    grad = np.zeros_like(centers)
    
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_soft_min_wall_dx = -exp_wall * signs / sum_wall
        
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        grad = grad_combined
    
    return -soft_min_combined, -grad.flatten()

def _soft_min_optimize(init_centers, temperature=0.001, max_iter=3000, time_limit=10.0):
    """Optimize using soft-min objective."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        return _soft_min_objective_and_grad(x, temperature)
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'maxfun': 8000}
    )
    
    return result.x.reshape(N, 3)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Iterative radius increase with energy minimization
    # Start from a small radius and increase until infeasible
    r_start = 0.08
    r_end = 0.115
    r_steps = 25
    
    r_values = np.linspace(r_start, r_end, r_steps)
    
    current_centers = grid.copy()
    current_r = r_start
    
    for i, r in enumerate(r_values):
        if time.time() > deadline - 60.0:
            break
        
        time_remaining = deadline - time.time() - 50.0
        if time_remaining < 3.0:
            break
        
        # Try to find a feasible configuration for radius r
        # Use multiple initializations
        inits = [current_centers]
        if i > 0 and i % 5 == 0:
            inits.append(_jitter_init(N, seed=i, jitter=0.01))
        if i > 0 and i % 10 == 0:
            inits.append(_random_init(N, seed=i + 100))
        
        best_for_r = None
        best_energy = float('inf')
        
        for init in inits:
            if time.time() > deadline - 50.0:
                break
            
            try:
                time_left = deadline - time.time() - 45.0
                if time_left < 2.0:
                    break
                
                optimized = _minimize_energy(init, r, max_iter=1500, time_limit=min(8.0, time_left))
                energy, _ = _energy_and_gradient(optimized.flatten(), r)
                
                if energy < best_energy:
                    best_energy = energy
                    best_for_r = optimized.copy()
            except Exception:
                continue
        
        if best_for_r is not None and best_energy < 1e-6:
            # Feasible! Update current state
            current_centers = best_for_r
            current_r = r
            r_check = _compute_radius(current_centers)
            if r_check > best_radius:
                best_radius = r_check
                best_centers = current_centers.copy()
        else:
            # Not feasible at this r, try a slightly smaller r
            if i > 0:
                # Go back to previous r and try harder
                r_prev = r_values[i-1]
                try:
                    time_left = deadline - time.time() - 40.0
                    if time_left > 3.0:
                        optimized = _minimize_energy(current_centers, r_prev, max_iter=3000, time_limit=min(10.0, time_left))
                        r_check = _compute_radius(optimized)
                        if r_check > best_radius:
                            best_radius = r_check
                            best_centers = optimized.copy()
                            current_centers = optimized
                except Exception:
                    pass
            break
    
    # Stage 2: Fine polishing with soft-min from best found
    if time.time() < deadline - 25.0:
        fine_temps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        for temp in fine_temps:
            if time.time() > deadline - 15.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 10.0
                if time_remaining < 3.0:
                    break
                optimized = _soft_min_optimize(best_centers, temperature=temp,
                                                max_iter=4000, time_limit=min(8.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 3: Final high-precision polish using L-BFGS with very small temperature
    if time.time() < deadline - 10.0:
        try:
            time_remaining = deadline - time.time() - 5.0
            if time_remaining > 3.0:
                optimized = _soft_min_optimize(best_centers, temperature=0.00001,
                                                max_iter=6000, time_limit=min(5.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
        except Exception:
            pass
    
    # Stage 4: Exact local polish on critical pairs
    if time.time() < deadline - 4.0:
        try:
            time_remaining = deadline - time.time() - 2.0
            if time_remaining > 1.0:
                polished = _polish_critical_pairs(best_centers, time_limit=min(2.5, time_remaining))
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
