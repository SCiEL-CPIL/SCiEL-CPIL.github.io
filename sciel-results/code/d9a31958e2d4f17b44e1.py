import numpy as np
import time

N = 50

def _score(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check bounds
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0

    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute distance to walls
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _violation_grad(points, r):
    """
    Compute gradient of the violation function.
    Violation = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - w_i)^2
    """
    N = len(points)
    grad = np.zeros_like(points)
    
    # Pairwise forces
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 0) # Avoid div by zero, handled by mask
    
    # Mask for pairs closer than 2r
    mask = dists < 2.0 * r
    # Ensure we don't divide by zero in gradient calc if dist is 0
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # Force magnitude: d/dr (max(0, 2r - d)^2) = 2 * (2r - d) * 2 = 4(2r - d)
    # Gradient w.r.t position i: -4(2r - d) * (p_i - p_j) / d
    # Sum over j
    
    # Create a matrix of force contributions
    # F_ij = 4 * (2r - d_ij) / d_ij * (p_i - p_j)
    # We only care about i != j and d_ij < 2r
    
    force_mag = np.zeros_like(dists)
    force_mag[mask] = 4.0 * (2.0 * r - dists[mask]) / safe_dists[mask]
    
    # grad_i = - sum_j force_mag[i,j] * (p_i - p_j)
    # Note: force_mag is symmetric.
    # Let's compute sum of vectors
    # diff[i,j] = p_i - p_j
    # grad[i] = - sum_j force_mag[i,j] * diff[i,j]
    
    # Using matrix multiplication or einsum
    # grad = - np.einsum('ij,ijk->ik', force_mag, diff)
    # But force_mag is 0 on diagonal.
    
    # Alternative efficient way:
    # grad[i] = sum_j 4(2r - d_ij)/d_ij * (p_j - p_i)
    # Let V = force_mag[:, :, None] * diff # (N, N, 2)
    # grad = -np.sum(V, axis=1)
    
    V = force_mag[:, :, np.newaxis] * diff
    grad = -np.sum(V, axis=1)
    
    # Wall forces
    # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # If x_i > 0: w_i depends on x_i. d(0.5-x)/dx = -1.
    # If x_i < 0: w_i depends on x_i. d(0.5+x)/dx = 1.
    # Violation term: max(0, r - w_i)^2.
    # Let v_i = r - w_i. If v_i > 0, grad += 2*v_i * grad(w_i)
    # grad(w_i) w.r.t x_i is -sign(x_i) if 0.5-|x_i| is the min, else 0.
    
    abs_x = np.abs(points[:, 0])
    abs_y = np.abs(points[:, 1])
    dx_wall = 0.5 - abs_x
    dy_wall = 0.5 - abs_y
    
    # Determine which wall is limiting for each point
    # If dx_wall < dy_wall, x is limiting. Else y is limiting (or both).
    
    # x contribution
    x_viol = r - dx_wall
    x_active = x_viol > 0
    # grad_x = -sign(x) * 2 * x_viol
    grad[:, 0] += -np.sign(points[:, 0]) * 2.0 * np.where(x_active, x_viol, 0.0)
    
    # y contribution
    y_viol = r - dy_wall
    y_active = y_viol > 0
    grad[:, 1] += -np.sign(points[:, 1]) * 2.0 * np.where(y_active, y_viol, 0.0)
    
    return grad

def _optimize_for_radius(r, init_points, start_time, deadline, max_iters=10000):
    """
    Try to find a configuration with radius r.
    Returns (success, points)
    """
    points = init_points.copy()
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        grad = _violation_grad(points, r)
        
        # Momentum update
        v = momentum * v - lr * grad
        points = points + v
        
        # Project to square [-0.5, 0.5]
        points = np.clip(points, -0.5, 0.5)
        
        # Check violation
        # Fast check: min dist and min wall dist
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r >= r:
            return True, points
            
        # Adaptive learning rate? Keep simple for now.
        if i % 1000 == 0 and i > 0:
            lr *= 0.9
            
    return False, points

def _grid_seed(n=7):
    ticks = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _hex_seed():
    # Generate hexagonal packing
    points = []
    # Approximate hex grid
    # Side length of hex triangle is 2r. 
    # We want to fit 50 points.
    # Let's just generate a large hex grid and take the first 50 that fit in a box
    # Or generate a specific pattern.
    # Simpler: Jittered grid is often good.
    # Let's use a jittered grid for diversity.
    n = 7
    ticks = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Jitter
    grid += np.random.normal(0, 0.01, grid.shape)
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_points = _grid_seed()
    best_score = _score(best_points)
    
    # Binary search on radius
    # Lower bound: 0.0
    # Upper bound: 0.5 / sqrt(50) approx? 
    # Max possible r is bounded by area: 50 * pi * r^2 <= 1 => r <= 1/sqrt(50*pi) ~ 0.08
    # Best known ~ 0.0714
    low = 0.0
    high = 0.08
    
    # We will try to find the max r.
    # Instead of pure binary search which might fail to converge in limited time,
    # we can do a few steps of binary search, or just optimize for a few candidate radii.
    # Better: Binary search with limited iterations.
    
    # Let's use a more robust approach: 
    # 1. Try to optimize for a radius slightly higher than current best.
    # 2. If success, increase low.
    # 3. If fail, decrease high.
    
    # Initial guess based on area
    r_guess = 0.07
    
    # We will run several restarts with different inits and try to improve best_score.
    # To be safe and fast, we will use a simplified binary search loop.
    
    # List of initializations
    inits = []
    inits.append(_grid_seed())
    inits.append(_hex_seed())
    inits.append(np.random.uniform(-0.4, 0.4, (N, 2)))
    inits.append(np.random.uniform(-0.4, 0.4, (N, 2)))
    
    # Binary search parameters
    num_steps = 10 # Number of binary search steps
    for step in range(num_steps):
        if time.time() > deadline - 5.0:
            break
            
        mid = (low + high) / 2.0
        
        # Try to find a config for 'mid'
        found = False
        for init in inits:
            if time.time() > deadline - 5.0:
                break
                
            # Allocate time for this optimization attempt
            # Roughly (deadline - now) / remaining_attempts
            success, points = _optimize_for_radius(mid, init, start_time, deadline)
            
            if success:
                found = True
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                low = mid
                break # If found, this radius is feasible, try higher
            else:
                # If not found, it might be infeasible, or we just didn't find it.
                # In binary search, we usually assume infeasible if optimization fails.
                # But to be robust, we might want to retry with different init if time permits.
                pass
        
        if not found:
            high = mid
            
    return best_points
