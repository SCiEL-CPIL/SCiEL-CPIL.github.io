"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check cube boundaries
    max_coord = np.max(np.abs(centers), axis=1)
    wall_dist = 0.5 - max_coord
    if np.any(wall_dist < -1e-10):
        return 0.0
    wall_dist = np.clip(wall_dist, 0, None)
    
    # Compute pairwise distances
    # Use a more memory-efficient approach for 100 points
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    r = min(min_pair_dist / 2.0, np.min(wall_dist))
    return max(r, 0.0)


def _objective_and_grad(centers, alpha=1.0):
    """
    Objective: maximize minimum distance to walls and between spheres.
    We minimize a smooth approximation: -min_dist + penalty for violations.
    Actually, we want to maximize the minimum of all pairwise distances and wall distances.
    Use a soft-min (log-sum-exp) approximation for the minimum.
    
    We'll use: minimize -logsumexp(alpha * dists) which approximates -min(dists)
    plus a hard penalty if any sphere is outside the cube.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    
    # Wall distances: for each sphere, distance to nearest wall
    # dist to wall in each direction
    wall_dists = np.empty((n, 6))
    for k in range(3):
        wall_dists[:, k] = 0.5 - centers[:, k]   # distance to +x wall
        wall_dists[:, k+3] = 0.5 + centers[:, k]  # distance to -x wall
    # The relevant wall distance for sphere i is min over its 6 wall distances
    min_wall = np.min(wall_dists, axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Combine all distances: wall distances and pairwise distances
    # We want to maximize the minimum of all these
    # Use log-sum-exp trick for soft minimum
    # soft_min = -logsumexp(-alpha * d) / alpha
    # But we have many distances. Let's collect them.
    
    # For efficiency, let's just compute the objective as:
    # f = -soft_min(all_dists) where all_dists includes wall and pair distances
    # soft_min(d) = -1/alpha * log(sum(exp(-alpha*d_i)))
    # So f = 1/alpha * log(sum(exp(-alpha*d_i)))
    
    # Collect all distances
    # Wall distances: n*6 values
    # Pair distances: n*(n-1)/2 values
    all_dists = np.concatenate([min_wall, dist[~np.eye(n, dtype=bool)]])
    
    # Clip to avoid overflow
    all_dists = np.clip(all_dists, -100, 100)
    
    # soft min
    max_d = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-alpha * (all_dists - max_d)))) + (-alpha * max_d)
    soft_min_val = -log_sum / alpha
    
    objective = -soft_min_val  # we minimize this, so we want to maximize soft_min
    
    # Gradient with respect to centers
    # d(soft_min)/d(centers) 
    # soft_min = -1/alpha * log(sum_j exp(-alpha * d_j))
    # d(soft_min)/d(c_i) = -1/alpha * sum_j [d(exp(-alpha*d_j))/dc_i / exp(-alpha*d_j)] * exp(-alpha*d_j) / sum...
    # = sum_j [alpha * exp(-alpha*d_j) * d(d_j)/dc_i] / sum_k exp(-alpha*d_k)
    # Wait, let me redo this.
    
    # Let S = sum_j exp(-alpha * d_j)
    # soft_min = -1/alpha * log(S)
    # d(soft_min)/dc_i = -1/alpha * (1/S) * dS/dc_i
    # dS/dc_i = sum_j exp(-alpha*d_j) * (-alpha) * d(d_j)/dc_i
    # So d(soft_min)/dc_i = -1/alpha * (1/S) * sum_j [-alpha * exp(-alpha*d_j) * d(d_j)/dc_i]
    # = sum_j [exp(-alpha*d_j) * d(d_j)/dc_i] / S
    # = sum_j [w_j * d(d_j)/dc_i] where w_j = exp(-alpha*d_j)/S
    
    # Gradient of objective = -gradient of soft_min
    # So grad_obj = -sum_j [w_j * d(d_j)/dc_i]
    
    # Compute weights
    exp_vals = np.exp(-alpha * (all_dists - max_d))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Wall distance gradients
    # For each sphere i, the min wall distance is min of 6 values
    # But we used min_wall which is the min over 6 directions
    # The gradient of min is tricky; let's use the soft approach consistently
    # Actually, for wall distances, let's compute gradient properly
    
    # Let's recompute with explicit handling
    # For wall distances, for sphere i, the relevant distance is min over 6 walls
    # d(min_wall_i)/dc_i[k] = -1 if the min is in the +k direction, +1 if in -k direction, 0 otherwise
    # But for a soft version, we need weights
    
    # Let me just compute the gradient numerically for simplicity if analytical is complex
    # But 100 points, 300 dimensions, numerical gradient is 300 function evaluations - too slow
    
    # Let's do analytical for pairwise and use a simple approach for walls
    
    # Actually, let's just use the pairwise gradient and add a wall penalty gradient
    
    # Pairwise gradient contribution
    # For each pair (i,j), d_ij = |c_i - c_j|
    # d(d_ij)/dc_i = (c_i - c_j)/d_ij
    # d(d_ij)/dc_j = (c_j - c_i)/d_ij
    
    # The weight for pair (i,j) is w_pair = exp(-alpha*d_ij)/S
    # But S includes wall distances too, so we need to be careful
    
    # Let me restructure: separate wall and pair contributions
    
    # Wall part
    wall_exp = np.exp(-alpha * (min_wall - max_d))
    wall_S = np.sum(wall_exp)
    
    # Pair part  
    pair_dists = dist[~np.eye(n, dtype=bool)]
    pair_exp = np.exp(-alpha * (pair_dists - max_d))
    pair_S = np.sum(pair_exp)
    
    S_total = wall_S + pair_S
    
    # Wall gradient: for each sphere i, the gradient of min_wall_i
    # Using soft: grad_wall_i = sum over 6 walls of [w_wall * grad(d_wall)]
    # where w_wall = exp(-alpha*d_wall)/S_total for that specific wall distance
    
    # This is getting complex. Let me just use a simpler objective.
    
    # Alternative: use a different formulation
    # Minimize: sum of squared violations + lambda * (-min_dist)
    # Or just use scipy minimize with a simple objective
    
    return objective, None  # placeholder


def _gradient_ascent_step(centers, lr=0.01, max_iter=1000):
    """Perform gradient ascent to maximize the minimum distance."""
    centers = centers.copy()
    
    for _ in range(max_iter):
        # Compute gradient numerically (central difference)
        eps = 1e-7
        grad = np.zeros_like(centers)
        for i in range(centers.shape[0]):
            for k in range(3):
                c_plus = centers.copy()
                c_plus[i, k] += eps
                c_minus = centers.copy()
                c_minus[i, k] -= eps
                f_plus = _compute_radius(c_plus)
                f_minus = _compute_radius(c_minus)
                grad[i, k] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent
        centers = centers + lr * grad
        
        # Project back to cube
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers


def _local_search(centers, alpha_schedule=None, max_time=60.0):
    """
    Two-phase local search:
    Phase 1: Coarse optimization with larger alpha (less smooth)
    Phase 2: Fine optimization with smaller alpha (more smooth)
    Phase 3: Polish - equalize binding constraints
    """
    if alpha_schedule is None:
        alpha_schedule = [(30, 150, 500), (100, 200, 300)]
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    start_time = time.time()
    
    for phase, (alpha_start, alpha_end, max_iter) in enumerate(alpha_schedule):
        for it in range(max_iter):
            if time.time() - start_time > max_time:
                break
            
            alpha = alpha_start + (alpha_end - alpha_start) * (it / max_iter)
            
            # Compute objective and gradient
            # Use a smooth objective
            n = centers.shape[0]
            
            # Wall distances
            min_wall = np.min(np.stack([
                0.5 - centers[:, 0], 0.5 + centers[:, 0],
                0.5 - centers[:, 1], 0.5 + centers[:, 1],
                0.5 - centers[:, 2], 0.5 + centers[:, 2]
            ], axis=1), axis=1)
            
            # Pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # All distances
            all_dists = np.concatenate([min_wall, dist[~np.eye(n, dtype=bool)]])
            
            # Soft min using log-sum-exp
            max_d = np.max(all_dists)
            log_sum = np.log(np.sum(np.exp(-alpha * (all_dists - max_d)))) - alpha * max_d
            soft_min = -log_sum / alpha
            
            # We want to maximize soft_min, so minimize -soft_min
            # Gradient of soft_min w.r.t. centers
            # As derived: grad(soft_min) = sum_j [w_j * grad(d_j)] where w_j = exp(-alpha*d_j)/S
            
            exp_vals = np.exp(-alpha * (all_dists - max_d))
            S = np.sum(exp_vals)
            weights = exp_vals / S
            
            grad = np.zeros_like(centers)
            
            # Wall contributions
            # For each sphere i, 6 wall distances
            # The weight for each wall distance is in the first n*6 entries of weights
            # But we used min_wall, so we need to track which wall is the min
            # For simplicity, let's compute gradient of each wall distance separately
            
            # Actually, let's just use a simpler approach: 
            # gradient of min_wall_i is the gradient of the active constraint
            # For a soft version, average over all walls with weights
            
            # Let me compute wall gradients properly
            wall_grads = np.zeros_like(centers)
            for i in range(n):
                for k in range(3):
                    # +k wall: d = 0.5 - c[i,k], grad = -1 in direction k
                    # -k wall: d = 0.5 + c[i,k], grad = +1 in direction k
                    # We need weights for each of the 6 wall distances
                    # But we only have min_wall, not individual weights
                    # Let's approximate: assume the min wall dominates
                    min_val = min_wall[i]
                    # Find which wall(s) are close to min
                    for k in range(3):
                        if abs((0.5 - centers[i, k]) - min_val) < 1e-6:
                            wall_grads[i, k] -= 1.0
                        if abs((0.5 + centers[i, k]) - min_val) < 1e-6:
                            wall_grads[i, k] += 1.0
            
            # Normalize wall gradients (they should sum to 1 in magnitude for the active constraint)
            # Actually, for a single active constraint, the gradient has magnitude 1
            # For multiple, it's a convex combination
            
            # Pairwise contributions
            for i in range(n):
                for j in range(i+1, n):
                    d_ij = dist[i, j]
                    if d_ij < 1e-10:
                        continue
                    # Weight for this pair
                    # The pair distance appears twice in all_dists (once for (i,j) and once for (j,i))
                    # But we used dist[~np.eye(n, dtype=bool)] which gives each pair once
                    # So we need to find the weight
                    # This is getting too complex. Let me use a simpler method.
                    pass
            
            # This analytical gradient is too complex. Let me use numerical gradient instead.
            # But that's slow. Let me use a different approach: use scipy's minimize.
            
            break  # Skip this phase for now, use scipy
    
    # Use scipy minimize with a simple objective
    from scipy.optimize import minimize
    
    def objective(x):
        c = x.reshape(-1, 3)
        r = _compute_radius(c)
        return -r  # minimize negative radius
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    start_time = time.time()
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() - start_time > max_time - 5:
            break
        
        # Initialize
        if restart == 0:
            x0 = centers.copy().flatten()
        else:
            # Perturb best found so far
            x0 = best_centers.copy()
            noise = rng.normal(0, 0.02, size=x0.shape)
            x0 = x0 + noise
            x0 = np.clip(x0, -0.49, 0.49)
        
        # Optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             bounds=[(-0.49, 0.49)] * (N * 3),
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            c = result.x.reshape(-1, 3)
            c = np.clip(c, -0.49, 0.49)
            r = _compute_radius(c)
            
            if r > best_radius:
                best_radius = r
                best_centers = c.copy()
        except Exception:
            pass
    
    return best_centers, best_radius


def _polish(centers, max_iter=200):
    """Polish by equalizing binding constraints."""
    centers = centers.copy()
    
    for _ in range(max_iter):
        # Compute all distances
        n = centers.shape[0]
        
        # Wall distances
        min_wall = np.min(np.stack([
            0.5 - centers[:, 0], 0.5 + centers[:, 0],
            0.5 - centers[:, 1], 0.5 + centers[:, 1],
            0.5 - centers[:, 2], 0.5 + centers[:, 2]
        ], axis=1), axis=1)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance
        min_pair = np.min(dist)
        min_wall_val = np.min(min_wall)
        
        # The binding constraint is the smaller of min_pair/2 and min_wall_val
        # Move spheres to increase the binding constraint
        
        # For simplicity, apply a small random perturbation to spheres involved in binding constraints
        # and keep if it improves
        
        r = _compute_radius(centers)
        
        # Find spheres with wall distance close to min
        wall_close = min_wall < min_wall_val + 1e-4
        # Find pairs with distance close to min
        pair_close = dist < min_pair + 1e-4
        
        # Perturb these spheres
        for i in range(n):
            if wall_close[i] or np.any(pair_close[i]):
                perturbation = np.random.normal(0, 0.001, size=3)
                new_c = centers.copy()
                new_c[i] += perturbation
                new_c = np.clip(new_c, -0.49, 0.49)
                new_r = _compute_radius(new_c)
                if new_r > r:
                    centers = new_c
                    r = new_r
    
    return centers


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    # Initial configuration: random points in the cube
    rng = np.random.default_rng(123)
    
    best_centers = None
    best_radius = 0.0
    
    # Start with a grid-like initialization for better starting point
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    initial_centers += rng.normal(0, 0.01, size=initial_centers.shape)
    initial_centers = np.clip(initial_centers, -0.49, 0.49)
    
    # Run local search
    centers, radius = _local_search(initial_centers, max_time=120.0)
    
    if radius > best_radius:
        best_radius = radius
        best_centers = centers.copy()
    
    # Polish if time permits
    if time.time() - start_time < deadline - 5:
        polished = _polish(best_centers, max_iter=100)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished
    
    # Final check and return
    if best_centers is None:
        return initial_centers
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
