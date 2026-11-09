import numpy as np
import time

N = 200

def compute_radius(points):
    # points: (N, 2)
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return r

def hex_init(rows):
    """
    Generate a hexagonal grid initialization with a specific number of rows.
    Returns an array of shape (N, 2) scaled to fit in [-0.5, 0.5].
    """
    if rows <= 0:
        return np.random.rand(N, 2) - 0.5
    
    cols = int(np.ceil(N / rows))
    # Adjust rows/cols to get exactly N
    # Fill row by row
    pts = []
    y_spacing = 1.0 / (rows * np.sqrt(3)/2 + 1) # Approximate scale
    # Better: generate raw coords, then scale
    
    raw_pts = []
    for r in range(rows):
        c_count = cols if r < (N - r * cols) else (N - r * cols)
        # This logic is tricky. Let's just make a grid of size rows x cols and take first N
        # Or distribute evenly.
        # Simple approach:
        pass
        
    # Standard hex grid generation
    pts = []
    # We want to fit N points.
    # Let's generate a grid of size rows x cols where rows*cols >= N
    # Then take the first N? No, that's biased.
    # Let's create a grid that is roughly square in aspect ratio.
    
    # Try to find rows, cols such that rows*cols >= N and aspect ratio is close to 1
    # For hex, aspect ratio is roughly cols / (rows * sqrt(3)/2)
    
    # Let's just generate a grid of rows x cols and trim/pad?
    # Actually, for packing, we want to fill the space.
    # Let's generate a grid of size rows x cols, and if we have more, remove some.
    # If we have less, add more?
    
    # Simpler: Generate rows x cols grid.
    # If rows*cols > N, we take the first N. This might leave a gap.
    # To be fair, we should center the points.
    
    # Let's generate a grid of size rows x cols.
    xs = []
    ys = []
    for r in range(rows):
        x_offset = 0.5 if r % 2 == 1 else 0.0
        for c in range(cols):
            x = c + x_offset
            y = r * np.sqrt(3) / 2.0
            xs.append(x)
            ys.append(y)
            
    pts = np.column_stack((xs, ys))
    
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points near the center? Or just use what we have?
        # The function must return N points.
        # Let's add random points
        while len(pts) < N:
            pts = np.vstack([pts, np.array([np.random.rand()-0.5, np.random.rand()-0.5])])
            
    # Center and scale
    center = np.mean(pts, axis=0)
    pts -= center
    
    max_coord = np.max(np.abs(pts))
    # Scale to fit in [-0.45, 0.45] to allow for movement
    scale = 0.45 / max_coord if max_coord > 0 else 1.0
    pts *= scale
    
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_r = 0
    best_points = None
    
    inits = []
    
    # 1. Hexagonal inits with different aspect ratios
    # N=200. sqrt(200) ~ 14.
    for rows in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        pts = hex_init(rows)
        inits.append(pts)
        
    # 2. Random inits
    for _ in range(10):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # Optimization
    # We use a soft-min objective: F = -1/lambda * log(sum(exp(-lambda d)))
    # Gradient descent on F is equivalent to pushing points apart.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        
        # Annealing schedule for lambda
        # Start smooth, end sharp
        lambda_schedule = [10, 50, 100, 200, 500, 1000, 5000]
        
        for lambda_val in lambda_schedule:
            if time.time() > deadline:
                break
                
            # Determine steps based on time remaining and lambda
            # Higher lambda -> fewer steps, smaller lr
            if lambda_val < 100:
                steps = 500
                lr = 0.01
            elif lambda_val < 500:
                steps = 300
                lr = 0.005
            else:
                steps = 100
                lr = 0.001
                
            # Adjust steps if time is tight
            time_per_step = (deadline - time.time()) / (steps * len(lambda_schedule)) # Rough
            # Just run steps, but check time inside loop
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Compute distances
                diff = pts[:, np.newaxis, :] - points[np.newaxis, :, :] # BUG: points vs pts
                # Correct:
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=-1)
                np.fill_diagonal(dist_sq, np.inf)
                
                # Weights for pairs: exp(-lambda * dist_sq)
                # To avoid underflow/overflow, we can normalize or just use small lambda initially
                # With lambda up to 5000, exp(-5000 * 0.01) = exp(-50) ~ 1e-22. OK.
                w_pairs = np.exp(-lambda_val * dist_sq)
                w_pairs[~np.isfinite(w_pairs)] = 0
                
                sum_w_pairs = np.sum(w_pairs, axis=1)
                
                # Gradient from pairs
                # grad_i = sum_j w_ij * (-2 lambda * (p_i - p_j)) / sum_w_i
                term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
                grad_pairs = np.sum(term, axis=1)
                
                sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
                grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
                
                # Wall terms
                x = pts[:, 0]
                y = pts[:, 1]
                
                d_x_l = x + 0.5
                d_x_r = 0.5 - x
                d_y_b = y + 0.5
                d_y_t = 0.5 - y
                
                # Squared distances
                d_x_l_sq = d_x_l**2
                d_x_r_sq = d_x_r**2
                d_y_b_sq = d_y_b**2
                d_y_t_sq = d_y_t**2
                
                w_x_l = np.exp(-lambda_val * d_x_l_sq)
                w_x_r = np.exp(-lambda_val * d_x_r_sq)
                w_y_b = np.exp(-lambda_val * d_y_b_sq)
                w_y_t = np.exp(-lambda_val * d_y_t_sq)
                
                sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
                
                grad_x_walls = (w_x_l * (-2 * lambda_val * d_x_l) + w_x_r * (2 * lambda_val * d_x_r)) / sum_w_walls
                grad_y_walls = (w_y_b * (-2 * lambda_val * d_y_b) + w_y_t * (2 * lambda_val * d_y_t)) / sum_w_walls
                
                grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
                
                total_grad = grad_pairs + grad_walls
                
                # Update
                pts_new = pts - lr * total_grad
                
                # Clip to bounds
                pts_new = np.clip(pts_new, -0.5, 0.5)
                
                pts = pts_new
                
                # Occasionally check radius
                if step % 50 == 0:
                    r = compute_radius(pts)
                    if r > best_r:
                        best_r = r
                        best_points = pts.copy()
                        
                # Random jitter if stuck (optional, adds cost)
                # if step % 100 == 0 and lambda_val > 100:
                #     pts += np.random.randn(N, 2) * 0.005
                #     pts = np.clip(pts, -0.5, 0.5)
                    
    # Final check
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    else:
        # One last polish with high lambda
        if time.time() < deadline - 1:
            pts = best_points.copy()
            lambda_val = 10000
            lr = 0.0001
            for _ in range(50):
                if time.time() > deadline:
                    break
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=-1)
                np.fill_diagonal(dist_sq, np.inf)
                
                w_pairs = np.exp(-lambda_val * dist_sq)
                w_pairs[~np.isfinite(w_pairs)] = 0
                sum_w_pairs = np.sum(w_pairs, axis=1)
                
                term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
                grad_pairs = np.sum(term, axis=1)
                sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
                grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
                
                x = pts[:, 0]
                y = pts[:, 1]
                d_x_l = x + 0.5
                d_x_r = 0.5 - x
                d_y_b = y + 0.5
                d_y_t = 0.5 - y
                
                w_x_l = np.exp(-lambda_val * d_x_l**2)
                w_x_r = np.exp(-lambda_val * d_x_r**2)
                w_y_b = np.exp(-lambda_val * d_y_b**2)
                w_y_t = np.exp(-lambda_val * d_y_t**2)
                
                sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
                
                grad_x_walls = (w_x_l * (-2 * lambda_val * d_x_l) + w_x_r * (2 * lambda_val * d_x_r)) / sum_w_walls
                grad_y_walls = (w_y_b * (-2 * lambda_val * d_y_b) + w_y_t * (2 * lambda_val * d_y_t)) / sum_w_walls
                
                grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
                
                total_grad = grad_pairs + grad_walls
                
                pts_new = pts - lr * total_grad
                pts_new = np.clip(pts_new, -0.5, 0.5)
                pts = pts_new
                
            r = compute_radius(pts)
            if r > best_r:
                best_points = pts
                
    return best_points
