import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _soft_min_gradient(centers, temp=0.005):
    """
    Computes the gradient of a smoothed minimum distance objective.
    Objective: r = min( min_{i<j} d_ij/2, min_i d_wall_i )
    We use a soft-min approximation: 
    soft_min(x) ≈ -temp * log(sum(exp(-x_i/temp)))
    Gradient of soft_min with respect to x_i is: - (exp(-x_i/temp) / sum(exp(-x_j/temp)))
    Since we want to maximize r, we move in the direction of the gradient of x_i.
    The weight for each constraint is proportional to how close it is to the minimum.
    """
    n = len(centers)
    
    # 1. Compute all pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero in gradient calc
    dists = np.sqrt(dist_sq)
    
    # Constraint values for circle-circle: d_ij / 2
    # We only consider i < j to avoid double counting, but for gradient symmetry 
    # it's easier to work with the full matrix and divide by 2 later or just sum.
    # Let's work with the full matrix of constraints: c_ij = d_ij / 2
    c_cc = dists / 2.0
    np.fill_diagonal(c_cc, np.inf) # Ignore self
    
    # Constraint values for circle-wall: 
    # d_wall_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
    # This is a min of 4 terms per point.
    # To keep it simple and differentiable, we can treat each wall constraint separately.
    # However, the soft-min over the global minimum handles the "active" constraints naturally.
    
    # Let's construct a vector of all active constraint values.
    # This might be large: N^2/2 + 4N.
    # For N=200, N^2/2 = 20000. 4N = 800. Total ~20800.
    # This is manageable.
    
    # Actually, computing the full soft-min gradient explicitly might be heavy.
    # Alternative: Use the "top-k" active constraints approximation.
    # Or, compute the gradient of the soft-min directly using the matrix.
    
    # Let's stick to the matrix formulation for circle-circle and handle walls separately.
    
    # Gradient for circle-circle part:
    # We want to maximize min(d_ij/2).
    # Let u_ij = d_ij / 2.
    # grad_u = - (exp(-u_ij/temp) / Z) * grad_u_ij
    # grad_u_ij w.r.t center_i: (center_i - center_j) / (2 * d_ij)
    # grad_u_ij w.r.t center_j: (center_j - center_i) / (2 * d_ij)
    
    # Weights w_ij = exp(-u_ij / temp)
    # Note: u_ij is symmetric.
    
    # To avoid overflow in exp, subtract the min value from u before exp.
    # But u varies. Let's just use the formula.
    
    # Efficient computation:
    # w_ij = exp(-d_ij / (2*temp))
    # Note: d_ij is in dists.
    
    # Let's compute weights for all pairs.
    # We need to be careful with the diagonal.
    
    # Let's use a slightly different approach: 
    # Compute the "influence" of each pair.
    
    # We will compute the gradient contribution for each pair (i, j) with i < j.
    # And for each wall constraint.
    
    # For speed, let's use the full matrix and rely on symmetry.
    
    # 1. Circle-Circle Weights
    # u_ij = d_ij / 2
    # w_ij = exp(-u_ij / temp)
    # To prevent underflow/overflow issues, we can normalize by the max weight or just use it.
    # Since temp is small, weights for non-minimal pairs will be ~0.
    
    # Let's compute log_weights = -dists / (2 * temp)
    # Then weights = exp(log_weights - max_log_weight) ? 
    # Actually, we just need the relative weights for the gradient direction.
    # grad_r = sum_j w_ij * grad_u_ij
    
    # Let's compute the sum of weighted vectors.
    
    # Vectorized:
    # diff is (N, N, 2)
    # dists is (N, N)
    
    # w_ij = exp(-dists / (2 * temp))
    # We set diagonal to 0.
    
    # To avoid numerical issues, let's find the min distance d_min.
    # d_min = min(dists[upper_triangle])
    # Then shift: u_shifted = u - d_min/2
    # Then w = exp(-u_shifted / temp)
    # This ensures the max weight is 1.
    
    # Get min dist (excluding diagonal)
    d_min_cc = np.min(dists[~np.eye(n, dtype=bool)])
    
    # Shifted distances
    # u_shifted = (dists - d_min_cc) / 2
    # w = exp(-u_shifted / temp)
    
    # Only compute for pairs that are "close" to the minimum to save time?
    # With N=200, full matrix is 40k entries. Exp is fast.
    
    u_shifted = (dists - d_min_cc) / 2.0
    np.fill_diagonal(u_shifted, np.inf)
    
    weights_cc = np.exp(-u_shifted / temp)
    np.fill_diagonal(weights_cc, 0.0)
    
    # Gradient of u_ij w.r.t center_i is (center_i - center_j) / (2 * d_ij)
    # Note: d_ij is in dists.
    # grad_vec_ij = diff / (2 * dists)
    # But we need to handle d=0. We set diagonal to 1 in dists earlier, so it's safe.
    
    # grad_cc_i = sum_j weights_cc[i, j] * (center_i - center_j) / (2 * d_ij)
    
    # Let's compute the term (center_i - center_j) / (2 * d_ij)
    # diff / (2 * dists[..., np.newaxis])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_dirs_cc = diff / (2.0 * dists[:, :, np.newaxis])
        
    # Replace inf/nan with 0
    grad_dirs_cc = np.nan_to_num(grad_dirs_cc, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Weighted sum
    # grad_cc = weights_cc @ grad_dirs_cc
    # Wait, grad_dirs_cc[i, j] is the gradient of u_ij w.r.t center_i?
    # u_ij = ||c_i - c_j|| / 2
    # d(u_ij)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # Yes, diff[i, j] is c_i - c_j.
    
    # So the gradient contribution to center_i from all j is:
    # sum_j w_ij * (c_i - c_j) / (2 * d_ij)
    
    grad_cc = weights_cc @ grad_dirs_cc
    
    # 2. Circle-Wall Weights
    # Constraints:
    # c_x_left_i = 0.5 - x_i
    # c_x_right_i = x_i + 0.5
    # c_y_bottom_i = 0.5 - y_i
    # c_y_top_i = y_i + 0.5
    
    # We want to maximize min of these.
    # Let's combine all wall constraints into a single soft-min as well, or treat them separately.
    # Treating them separately is safer for direction.
    
    # Let's find the global min among all constraints (CC and CW) to determine the "active" set.
    # Actually, the soft-min gradient automatically handles the weighting.
    
    # For walls, let's compute the soft-min gradient similarly.
    # Values:
    val_x_left = 0.5 - centers[:, 0]
    val_x_right = centers[:, 0] + 0.5
    val_y_bottom = 0.5 - centers[:, 1]
    val_y_top = centers[:, 1] + 0.5
    
    # Stack them into a matrix of shape (N, 4)
    wall_vals = np.stack([val_x_left, val_x_right, val_y_bottom, val_y_top], axis=1)
    
    # Find min wall value
    d_min_cw = np.min(wall_vals)
    
    # Shift
    wall_vals_shifted = wall_vals - d_min_cw
    wall_weights = np.exp(-wall_vals_shifted / temp)
    
    # Gradient of each wall constraint w.r.t center:
    # d(0.5 - x)/dx = -1
    # d(x + 0.5)/dx = 1
    # d(0.5 - y)/dy = -1
    # d(y + 0.5)/dy = 1
    
    # grad_wall_i = 
    #   w_x_left * (-1, 0) + w_x_right * (1, 0) + w_y_bottom * (0, -1) + w_y_top * (0, 1)
    
    grad_wall = np.zeros((n, 2))
    grad_wall[:, 0] = -wall_weights[:, 0] + wall_weights[:, 1]
    grad_wall[:, 1] = -wall_weights[:, 2] + wall_weights[:, 3]
    
    # Combine gradients
    # The total gradient is the sum of the weighted gradients.
    # Note: The scaling of the soft-min for CC and CW might be different if the mins are different.
    # Ideally, we should use a single temperature for the global minimum.
    # Let's refine: Use the global minimum of all constraints to shift.
    
    # Let's re-calculate with a global shift for better weighting balance.
    d_min_global = min(d_min_cc / 2.0, d_min_cw)
    
    # Re-calculate CC weights with global shift
    u_cc = dists / 2.0
    np.fill_diagonal(u_cc, np.inf)
    u_cc_shifted = u_cc - d_min_global
    weights_cc = np.exp(-u_cc_shifted / temp)
    np.fill_diagonal(weights_cc, 0.0)
    
    # Re-calculate CW weights with global shift
    wall_vals_shifted = wall_vals - d_min_global
    wall_weights = np.exp(-wall_vals_shifted / temp)
    
    # Recalculate gradients
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_dirs_cc = diff / (2.0 * dists[:, :, np.newaxis])
    grad_dirs_cc = np.nan_to_num(grad_dirs_cc, nan=0.0, posinf=0.0, neginf=0.0)
    
    grad_cc = weights_cc @ grad_dirs_cc
    
    grad_wall = np.zeros((n, 2))
    grad_wall[:, 0] = -wall_weights[:, 0] + wall_weights[:, 1]
    grad_wall[:, 1] = -wall_weights[:, 2] + wall_weights[:, 3]
    
    total_grad = grad_cc + grad_wall
    
    return total_grad

def _optimize_gradient(centers, time_limit, start_time, max_iters=5000, lr=0.01):
    """
    Gradient ascent on the smoothed min-radius objective.
    """
    centers = centers.copy()
    n = len(centers)
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    # Adaptive learning rate
    current_lr = lr
    stall_count = 0
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient
        grad = _soft_min_gradient(centers, temp=0.005)
        
        # Update
        centers += current_lr * grad
        
        # Project to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Evaluate every 10 steps
        if i % 10 == 0 or i == max_iters - 1:
            current_score = _compute_score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 50:
                    current_lr *= 0.8
                    if current_lr < 1e-6:
                        break
                    stall_count = 0
                    
    return best_centers, best_score

def _generate_hex_grid(N):
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    inits = []
    
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Run gradient optimization on each initialization
    for score, init_centers in scored_inits:
        if time.time() - start_time > time_limit:
            break
            
        centers, local_score = _optimize_gradient(init_centers, time_limit=time_limit, start_time=start_time, max_iters=2000, lr=0.02)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = centers
            
    # If we have time, try more random restarts from the best solution
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(int(time.time() * 1000) % 10000)
            perturbed = best_centers + np.random.normal(0, 0.01, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            centers, local_score = _optimize_gradient(perturbed, time_limit=time_limit, start_time=start_time, max_iters=1000, lr=0.01)
            
            if local_score > best_score:
                best_score = local_score
                best_centers = centers
                
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
