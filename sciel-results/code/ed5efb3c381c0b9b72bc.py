import numpy as np
import time

N = 50

def compute_radius_fast(positions, moved_idx=None):
    """
    Compute the maximum radius for the given positions.
    If moved_idx is provided, only recompute distances involving that point.
    Returns (radius, is_valid).
    """
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    if np.any(bound_dists < 0):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    if moved_idx is not None:
        # Only compute distances from moved point to others
        diff = positions[moved_idx] - positions
        dists_sq = np.sum(diff**2, axis=-1)
        # Exclude self
        dists_sq[moved_idx] = np.inf
        min_dist_sq = np.min(dists_sq)
    else:
        # Full pairwise
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
    
    min_dist = np.sqrt(min_dist_sq)
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    If invalid (points outside), radius is -1.
    """
    return compute_radius_fast(positions, moved_idx=None)

def soft_min_surrogate(positions, beta=50.0):
    """
    Smooth surrogate for the min distance objective using log-sum-exp.
    Returns a smooth approximation of min(min_dist/2, min_bound).
    """
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft min of pairwise distances
    # Use log-sum-exp for stability
    min_dists = np.sqrt(dists_sq)
    min_dists_flat = min_dists.flatten()
    min_dists_flat = min_dists_flat[~np.isinf(min_dists_flat)]
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta*x)))/beta approximates min(x)
    # For numerical stability, subtract max
    x = min_dists_flat / 2.0  # These are candidate radii from pairs
    b = beta
    x_max = np.max(x)
    log_sum = x_max + np.log(np.sum(np.exp(-b * (x - x_max))))
    soft_min_dist = -log_sum / b
    
    # Soft min of boundary distances
    x_bound = bound_dists
    x_bound_max = np.max(x_bound)
    log_sum_bound = x_bound_max + np.log(np.sum(np.exp(-b * (x_bound - x_bound_max))))
    soft_min_bound = -log_sum_bound / b
    
    # Combine: soft min of the two soft mins
    # Just take the min for simplicity, or soft min again
    # For gradient, we can use the smaller one with soft blending
    r = min(soft_min_dist, soft_min_bound)
    
    return r

def polish_gradient(positions, max_iter=2000, tol=1e-10):
    """
    Final polish using gradient ascent on the smooth surrogate.
    """
    beta = 100.0
    lr = 0.001
    positions = positions.copy()
    
    best_r, _ = compute_radius(positions)
    best_pos = positions.copy()
    
    for it in range(max_iter):
        # Compute gradient of soft_min_surrogate w.r.t. positions
        # Numerical gradient (central differences)
        grad = np.zeros_like(positions)
        eps = 1e-6
        
        for i in range(N):
            for k in range(2):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, k] += eps
                pos_minus[i, k] -= eps
                
                f_plus = soft_min_surrogate(pos_plus, beta)
                f_minus = soft_min_surrogate(pos_minus, beta)
                
                grad[i, k] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent
        positions += lr * grad
        
        # Clamp to stay inside
        positions = np.clip(positions, -0.4999, 0.4999)
        
        r, is_valid = compute_radius(positions)
        if is_valid and r > best_r:
            best_r = r
            best_pos = positions.copy()
        
        # Check convergence
        if it % 100 == 0 and it > 0:
            # If no improvement in last 100 iters, break
            pass
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
    
    # Add small noise to break symmetry
    r_init, _ = compute_radius(positions)
    noise_std = max(0.001, r_init * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius, _ = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.001
    T_end = 1e-8
    alpha = 0.9995
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    # Sliding window for adaptive step size
    window_size = 200
    recent_accepts = []
    no_improve_count = 0
    last_best = best_radius
    
    while time.time() < deadline:
        # Batch of steps
        batch_size = 50
        for _ in range(batch_size):
            if time.time() >= deadline:
                break
                
            # Pick random point
            idx = np.random.randint(N)
            
            # Generate move
            delta = np.random.normal(0, step_size, size=2)
            new_pos = positions[idx].copy() + delta
            
            old_pos = positions[idx].copy()
            positions[idx] = new_pos
            new_radius, is_valid = compute_radius_fast(positions, moved_idx=idx)
            
            if not is_valid:
                new_radius = -1.0
            
            # Metropolis Acceptance
            if new_radius > current_radius:
                current_radius = new_radius
                recent_accepts.append(1)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        recent_accepts.append(1)
                        no_improve_count += 1
                    else:
                        positions[idx] = old_pos
                        recent_accepts.append(0)
                        no_improve_count += 1
                else:
                    positions[idx] = old_pos
                    recent_accepts.append(0)
                    no_improve_count += 1
            
            # Adaptive step size based on sliding window
            if len(recent_accepts) >= window_size:
                accept_rate = np.mean(recent_accepts[-window_size:])
                if accept_rate < 0.15:
                    step_size *= 1.3
                elif accept_rate > 0.85:
                    step_size *= 0.7
                step_size = np.clip(step_size, min_step_size, max_step_size)
                # Keep window bounded
                if len(recent_accepts) > 2 * window_size:
                    recent_accepts = recent_accepts[-window_size:]
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # Restart if stuck
        if no_improve_count > 500 and time.time() < deadline - 30:
            # Perturb around best
            perturbation = 0.05
            positions = best_positions + np.random.normal(0, perturbation, size=best_positions.shape)
            positions = np.clip(positions, -0.4999, 0.4999)
            current_radius, _ = compute_radius(positions)
            no_improve_count = 0
            T = max(T, 0.0005)
    
    # 3. Final polish phase
    if time.time() < deadline - 5:
        polished, polished_r = polish_gradient(best_positions, max_iter=500, tol=1e-10)
        if polished_r > best_radius:
            best_radius = polished_r
            best_positions = polished
    
    return best_positions
