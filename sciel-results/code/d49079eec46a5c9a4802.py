import numpy as np
import time

N = 200

def compute_radius(points):
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_grid_init(n_rows, jitter=0.0):
    """Generate hexagonal lattice with optional jitter."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.45 / max_extent
        pts *= scale
    
    if jitter > 0:
        pts += np.random.randn(N, 2) * jitter
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def optimize(pts, deadline, lambda_start=20.0, lambda_end=200.0, n_steps=800, lr_start=0.002):
    """Multi-stage gradient optimization with adaptive lambda."""
    pts = pts.copy()
    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Adaptive lambda: start low for exploration, increase for exploitation
        progress = step / max(n_steps - 1, 1)
        lambda_val = lambda_start + (lambda_end - lambda_start) * progress
        
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Pairwise forces
        w_pairs = np.exp(-lambda_val * dist_sq)
        w_pairs[~np.isfinite(w_pairs)] = 0
        sum_w_pairs = np.sum(w_pairs, axis=1)
        
        term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
        grad_pairs = np.sum(term, axis=1)
        sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
        grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
        
        # Wall forces - smoother
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        # Use quadratic penalty with soft threshold
        wall_strength = 5.0
        w_x_l = np.exp(-lambda_val * d_x_l_sq if False else -lambda_val * np.maximum(0, 0.02 - d_x_l)**2)
        w_x_r = np.exp(-lambda_val * np.maximum(0, 0.02 - d_x_r)**2)
        w_y_b = np.exp(-lambda_val * np.maximum(0, 0.02 - d_y_b)**2)
        w_y_t = np.exp(-lambda_val * np.maximum(0, 0.02 - d_y_t)**2)
        
        # Simpler wall gradient: linear push when close to wall
        grad_x_walls = np.zeros_like(x)
        grad_y_walls = np.zeros_like(y)
        
        margin = 0.03
        grad_x_walls[d_x_l < margin] -= wall_strength * (margin - d_x_l[d_x_l < margin])
        grad_x_walls[d_x_r < margin] += wall_strength * (margin - d_x_r[d_x_r < margin])
        grad_y_walls[d_y_b < margin] -= wall_strength * (margin - d_y_b[d_y_b < margin])
        grad_y_walls[d_y_t < margin] += wall_strength * (margin - d_y_t[d_y_t < margin])
        
        grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
        
        total_grad = grad_pairs + grad_walls
        
        # Adaptive learning rate
        lr = lr_start / (1 + step * 0.002)
        
        pts_new = pts - lr * total_grad
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        # Occasionally add small noise to escape local minima
        if step % 150 == 75:
            pts_new += np.random.randn(N, 2) * 0.005
            pts_new = np.clip(pts_new, -0.5, 0.5)
        
        pts = pts_new
        
        # Check improvement periodically
        if step % 50 == 0:
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_r = 0
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal grids with various row counts
    for n_rows in [12, 13, 14, 15, 16]:
        inits.append(hex_grid_init(n_rows, jitter=0.0))
        inits.append(hex_grid_init(n_rows, jitter=0.01))
        inits.append(hex_grid_init(n_rows, jitter=0.02))
    
    # Random initializations
    for _ in range(10):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
    
    # Grid-based initialization
    m = 14
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    inits.append(grid_pts)
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        pts, r = optimize(init, deadline)
        
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    # Final polish on best solution
    if best_points is not None and time.time() < deadline - 5:
        # High-precision local optimization
        polish_deadline = min(deadline, time.time() + 10)
        best_points, best_r = optimize(best_points, polish_deadline, 
                                        lambda_start=100.0, lambda_end=500.0, 
                                        n_steps=500, lr_start=0.0005)
    
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
