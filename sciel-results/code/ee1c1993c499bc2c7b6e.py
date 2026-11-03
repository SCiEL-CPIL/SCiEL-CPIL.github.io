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

def _objective_and_grad(x, temperature=0.01):
    """
    Objective: maximize soft-min radius. Returns (-radius, gradient).
    Uses a soft-min surrogate with analytical gradients.
    """
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    # Pairwise soft-min of distances/2
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        # We want soft min of d/2 where d = sqrt(sq_d)
        # soft_min(d/2) ≈ sqrt(soft_min(d^2/4))
        # soft_min(d^2/4) = max_sq/4 - T2 * log(sum(exp(-(sq_d - max_sq)/(4*T2))))
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
    
    # Wall soft-min
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combined soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp  # (N, N) matrix
        sum_w = np.sum(weights, axis=1)  # (N,)
        weighted_sum_x = weights @ centers  # (N, 3)
        
        # d(soft_min_sq_pair)/d(centers[i]) = 4*(centers[i] * sum_w[i] - weighted_sum_x[i]) / 4
        # Actually: soft_min_sq_pair = max_sq/4 - T2*log(sum_exp)
        # d(soft_min_sq_pair)/d(c_i) = sum_j weights[i,j] * (c_i - c_j) * ... 
        # Let's derive properly:
        # soft_min_sq_pair = max_sq/4 - T2 * log(sum_j exp(-(sq_d_ij - max_sq)/(4T2)))
        # where sq_d_ij = |c_i - c_j|^2
        # d(sq_d_ij)/d(c_i) = 2(c_i - c_j)
        # d(soft_min_sq_pair)/d(c_i) = -T2 * sum_j [d/d(c_i) log(exp(-(sq_d_ij - max_sq)/(4T2)))]
        # = -T2 * sum_j [exp(-(sq_d_ij - max_sq)/(4T2)) / sum_exp * (-1/(4T2)) * 2(c_i - c_j)]
        # = -T2 * sum_j [weights[i,j] * (-1/(2T2)) * 2(c_i - c_j)]
        # = sum_j weights[i,j] * (c_i - c_j)
        # = sum_w[i] * c_i - weighted_sum_x[i]
        
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        # Now d(soft_min_pair)/d(c_i) = d(sqrt(soft_min_sq_pair))/d(c_i)
        # = grad_pair_sq / (2 * soft_min_pair)
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        # Wall gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        
        # d(soft_min_wall)/d(c_i) = -T2 * d/d(c_i) log(sum exp(-(wall_d - max_wall)/T2))
        # wall_d = 0.5 - |c_i|, d(wall_d)/d(c_i) = -sign(c_i)
        # d(soft_min_wall)/d(c_i) = -T2 * sum_j [exp_wall[j] / sum_wall * (-1/T2) * (-sign(c_j))]
        # = -T2 * sum_j [exp_wall[j] / sum_wall * (1/T2) * sign(c_j)]
        # = -sum_j [exp_wall[j] / sum_wall * sign(c_j)]
        # = -exp_wall / sum_wall * signs
        
        d_soft_min_wall_dx = -exp_wall * signs / sum_wall
        
        # Combined gradient
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        
        grad = grad_combined
    
    # We want to MAXIMIZE soft_min_combined, so minimize -soft_min_combined
    return -soft_min_combined, -grad

def _optimize_from_init(init_centers, temperature=0.01, max_iter=3000, time_limit=15.0):
    """Optimize from initial centers using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, temperature)
        return val, grad
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'maxfun': 10000}
    )
    
    return result.x.reshape(N, 3)

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Find the minimum radius constraint
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # If pair distance is limiting, push those two apart
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            # Push apart along the connecting line
            push = (r - dist/2.0) * d / dist * 0.5
            centers[i] += push
            centers[j] -= push
        else:
            # Wall is limiting, push that sphere inward
            i, k = min_wall_idx
            wall = wall_dists[i, k]
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        # Re-clip to cube
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

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
    # Start with grid and add body-centered points
    grid = _grid_init(n)
    rng = np.random.RandomState(42)
    # Add small random perturbations to break symmetry
    noise = rng.uniform(-0.005, 0.005, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Coarse optimization with larger temperature
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
    
    inits = [grid]
    # Add jittered grids to help escape symmetric local minima
    for seed in range(6):
        inits.append(_jitter_init(N, seed=seed, jitter=0.015))
    for seed in range(6):
        inits.append(_random_init(N, seed=seed))
    inits.append(_body_centered_init(N))
    
    for temp in temps:
        if time.time() > deadline - 45.0:
            break
        
        for init in inits:
            if time.time() > deadline - 35.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 30.0
                if time_remaining < 5.0:
                    break
                optimized = _optimize_from_init(init, temperature=temp, 
                                                max_iter=2500, time_limit=min(12.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update inits with best found so far
        inits = [best_centers] + inits[:3]
    
    # Stage 2: Fine polishing from best with very small temperature
    if time.time() < deadline - 20.0:
        fine_temps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        for temp in fine_temps:
            if time.time() > deadline - 10.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 8.0
                if time_remaining < 3.0:
                    break
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=4000, time_limit=min(8.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 3: Final high-precision polish using L-BFGS with very small temperature
    if time.time() < deadline - 8.0:
        try:
            time_remaining = deadline - time.time() - 4.0
            if time_remaining > 3.0:
                # Use a very small temperature to sharpen the soft-min approximation
                optimized = _optimize_from_init(best_centers, temperature=0.00001,
                                                max_iter=6000, time_limit=min(5.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
        except Exception:
            pass
    
    # Stage 4: Exact local polish on critical pairs
    if time.time() < deadline - 3.0:
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
