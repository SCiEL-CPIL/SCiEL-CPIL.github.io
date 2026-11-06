import numpy as np
import time

N = 200
DEADLINE = 145.0  # Safety margin before 150s limit

def _compute_radius(points):
    """
    Compute the maximum common radius for given circle centers.
    Returns 0.0 if invalid.
    """
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 0.5):
        return 0.0
    if np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 0.5):
        return 0.0

    # Center-center distances
    # Use cdist for efficiency if available, otherwise manual
    # cdist is from scipy.spatial.distance
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_center_dist = np.min(dists)
    except ImportError:
        # Fallback to manual calculation (slower but works)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_center_dist = np.sqrt(np.min(dist_sq))

    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))

    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    """Generate a hexagonal grid of points inside the unit square."""
    # Approximate number of columns and rows
    # Area of hexagon is approx sqrt(3)/2 * s^2
    # We want n points.
    # Let's just generate a grid and filter.
    cols = 16
    rows = 14
    # Adjust spacing to fit
    s = 1.0 / (cols + 0.5)
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
           (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > n:
        # Select n points greedily by max min-distance (Farthest Point Sampling)
        # To save time, we just take the first n, as the optimizer will fix it.
        # But a better init is better. Let's do a quick random selection if too many.
        # Actually, for N=200, the grid is roughly 16x14=224. 
        # We can just pick the most central ones or random.
        # Random is fine if we have a good optimizer.
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        missing = n - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (missing, 2))
        points = np.vstack([points, random_pts])
    
    return points

def _optimize_hinge(points, r_target, n_iters, start_time, deadline, lr=0.005, noise_scale=0.001):
    """
    Optimize points to maximize the minimum distance, using a hinge loss penalty
    for distances < 2*r_target.
    """
    points = points.copy().astype(np.float64)
    
    for it in range(n_iters):
        if time.time() > deadline - 2:
            break
        
        # Compute pairwise differences and distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        # Gradient of the loss function
        # Loss = sum_{i<j} max(0, 2*r_target - d_ij)^2 + sum_i max(0, r_target - w_i)^2
        # Note: We use 2*r_target for center-center because radius is d/2.
        # So we want d_ij >= 2*r_target.
        # Wall: we want w_i >= r_target.
        
        grad = np.zeros_like(points)
        
        # Pairwise gradients
        # Only for i < j
        triu_idx = np.triu_indices(N, k=1)
        d_ij = dist[triu_idx]
        
        # Hinge violation
        # v_ij = 2*r_target - d_ij
        # If v_ij > 0, loss = v_ij^2
        # d(loss)/d(d_ij) = -2 * v_ij
        # d(d_ij)/d(points[i]) = (points[i] - points[j]) / d_ij
        # So grad_points[i] += -2 * v_ij * (points[i] - points[j]) / d_ij
        # grad_points[j] += -2 * v_ij * (points[j] - points[i]) / d_ij
        
        v_ij = 2.0 * r_target - d_ij
        mask = v_ij > 0
        
        if np.any(mask):
            v_active = v_ij[mask]
            d_active = d_ij[mask]
            
            # Coefficient for gradient
            # -2 * v_ij / d_ij
            coeff = -2.0 * v_active / d_active
            
            i_idx, j_idx = triu_idx[0][mask], triu_idx[1][mask]
            
            # Vectorized update
            # grad[i] += coeff * (p[i] - p[j])
            # grad[j] -= coeff * (p[i] - p[j])
            
            # We can use np.add.at for accumulation
            np.add.at(grad, i_idx, coeff[:, np.newaxis] * (points[i_idx] - points[j_idx]))
            np.add.at(grad, j_idx, -coeff[:, np.newaxis] * (points[i_idx] - points[j_idx]))
            
        # Wall gradients
        # w_x = 0.5 - |x|
        # v_x = r_target - w_x
        # If v_x > 0, loss = v_x^2
        # d(loss)/d(w_x) = -2 * v_x
        # d(w_x)/d(x) = -sign(x)
        # grad_x += -2 * v_x * (-sign(x)) = 2 * v_x * sign(x)
        
        dx = 0.5 - np.abs(points[:, 0])
        dy = 0.5 - np.abs(points[:, 1])
        
        v_dx = r_target - dx
        v_dy = r_target - dy
        
        mask_dx = v_dx > 0
        if np.any(mask_dx):
            grad[:, 0] += 2.0 * v_dx[mask_dx] * np.sign(points[mask_dx, 0])
            
        mask_dy = v_dy > 0
        if np.any(mask_dy):
            grad[:, 1] += 2.0 * v_dy[mask_dy] * np.sign(points[mask_dy, 1])
            
        # Update points
        points -= lr * grad
        
        # Add noise to escape local minima
        if noise_scale > 0:
            points += np.random.normal(0, noise_scale, points.shape)
            
        # Clip to stay inside
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Decay learning rate and noise
        lr *= 0.999
        noise_scale *= 0.995
        
    return points

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Generate initial hex grid
    init_points = _generate_hex_grid(N)
    init_points = np.clip(init_points, -0.48, 0.48)
    
    best_points = init_points.copy()
    best_score = _compute_radius(init_points)
    
    # Annealing schedule for r_target
    # We want to reach ~0.0366
    # Start low, increase gradually
    
    r_current = 0.01
    r_max = 0.03662
    
    # Number of iterations per step
    # We have ~140 seconds. 
    # Each iteration is O(N^2). N=200 -> 40000 ops. Very fast in numpy.
    # Let's do 500 iterations per r step.
    n_iters_per_step = 500
    
    # We will loop until time runs out or r_max is reached
    # We also do "restarts" if stuck
    
    stuck_counter = 0
    last_improvement_r = best_score
    
    while time.time() < DEADLINE - 5:
        # Try to improve current best
        # Use a slightly higher r_target to push
        r_target = min(r_max, r_current * 1.05)
        
        # If we are close to best_score, try to push a bit more
        if r_target < best_score * 1.01:
             r_target = best_score * 1.02
        
        # Run optimization
        points = _optimize_hinge(best_points, r_target, n_iters_per_step, start_time, DEADLINE, lr=0.005, noise_scale=0.002)
        
        new_score = _compute_radius(points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            r_current = new_score
            stuck_counter = 0
        else:
            # No improvement
            stuck_counter += 1
            
            # If stuck for a while, restart with noise
            if stuck_counter > 5:
                # Add larger noise to best_points
                noisy_points = best_points + np.random.normal(0, 0.01, best_points.shape)
                noisy_points = np.clip(noisy_points, -0.49, 0.49)
                
                # Try to optimize the noisy version with a slightly lower target
                r_retry = max(0.01, best_score * 0.95)
                points = _optimize_hinge(noisy_points, r_retry, 200, start_time, DEADLINE, lr=0.005, noise_scale=0.01)
                retry_score = _compute_radius(points)
                
                if retry_score > best_score:
                    best_score = retry_score
                    best_points = points.copy()
                    r_current = best_score
                    stuck_counter = 0
                else:
                    # Reset r_current to allow trying again from current best
                    r_current = best_score
                    stuck_counter = 0
            
            # Reduce r_target slightly if stuck? No, keep pushing.
            # The optimizer should handle it.
            
    # Final polish: try to maximize the minimum distance directly using a small step size
    # Use the current best_score as the target, but with very small noise and lr
    if time.time() < DEADLINE - 2:
        final_points = _optimize_hinge(best_points, best_score, 500, start_time, DEADLINE, lr=0.001, noise_scale=0.0)
        final_score = _compute_radius(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points

    # Ensure all points are strictly inside
    best_points[:, 0] = np.clip(best_points[:, 0], -0.49999, 0.49999)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.49999, 0.49999)
    
    return best_points
