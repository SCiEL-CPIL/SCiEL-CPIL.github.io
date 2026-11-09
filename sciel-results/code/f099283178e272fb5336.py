import numpy as np
import time

N = 26

def _is_valid(circles):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    if n == 0:
        return True
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < (min_dist - 1e-7) ** 2:
                return False
    return True

def _score(circles):
    """Compute sum of radii if valid, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0

def _make_initial_grid():
    """Create initial grid layout."""
    # 5x6 grid is 30, we need 26.
    # Let's use a 5x6 grid and remove 4 positions (e.g., corners or specific spots)
    # Or just place 26 in a 5x6 grid, skipping 4 positions.
    
    rows, cols = 5, 6
    points = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            # Initial radius small
            r = 0.5 / max(rows, cols) * 0.5
            points.append([x, y, r])
            count += 1
        if count >= N:
            break
            
    return np.array(points)

def _make_initial_random(seed):
    """Create random initial layout."""
    np.random.seed(seed)
    circles = []
    for _ in range(N):
        r = np.random.uniform(0.02, 0.08)
        x = np.random.uniform(r, 1 - r)
        y = np.random.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)

def _make_initial_clustered(seed):
    """Create clustered initialization - some larger, some smaller."""
    np.random.seed(seed)
    xs = []
    ys = []
    rs = []
    # Place 5 "large" circles first
    for _ in range(5):
        x = np.random.uniform(0.2, 0.8)
        y = np.random.uniform(0.2, 0.8)
        r = np.random.uniform(0.08, 0.12)
        xs.append(x)
        ys.append(y)
        rs.append(r)
    # Place remaining 21 smaller circles
    for _ in range(N - 5):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.03, 0.06)
        xs.append(x)
        ys.append(y)
        rs.append(r)
        
    return np.array(list(zip(xs, ys, rs)))

def _post_process(circles, time_limit=None):
    """
    Ensure validity by slightly shrinking radii or moving centers if necessary.
    This is a safety net.
    """
    if time_limit is not None and time.time() > time_limit:
        # Emergency shrink to ensure validity if time is up
        r_max_bound = np.minimum(
            np.minimum(circles[:, 0], 1 - circles[:, 0]),
            np.minimum(circles[:, 1], 1 - circles[:, 1])
        )
        circles[:, 2] = np.minimum(circles[:, 2], r_max_bound * 0.9)
        return circles

    circles = circles.copy()
    # Check validity
    if _is_valid(circles):
        return circles
    
    # If invalid, shrink radii slightly
    # This is a crude fix, but ensures we return *something* valid if optimizer fails
    # Ideally, we want to avoid this.
    
    # Try to fix by reducing radii
    for _ in range(10):
        if _is_valid(circles):
            break
        circles[:, 2] *= 0.95 # Shrink radii by 5%
        
    # If still invalid, force valid positions
    if not _is_valid(circles):
        # Fallback to grid with very small radii
        grid = _make_initial_grid()
        grid[:, 2] = 0.01
        circles = grid
        
    return circles

def _gradient_descent(x, y, r, deadline):
    """
    Perform gradient descent on the penalty objective.
    Returns updated x, y, r arrays.
    """
    lr = 0.005
    lambda_overlap = 5.0
    lambda_bound = 5.0
    
    # Determine max steps based on remaining time
    elapsed = time.time() - (deadline - 115.0) # Approximate start
    # Better to just pass start time or estimate
    # Let's just run until deadline or max steps
    
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Calculate Distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero in gradients
        d_safe = np.where(d < 1e-8, 1e-8, d)
        
        # Overlap Penalty
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        np.fill_diagonal(overlap, 0)
        
        # Gradients for Overlap
        # d/dx_i sum_{j != i} max(0, r_i+r_j - d_ij)^2
        # = sum_{j != i} 2 * max(0, r_i+r_j - d_ij) * (- (x_i - x_j) / d_ij)
        # = -2 * sum_{j != i} overlap_ij * (x_i - x_j) / d_ij
        # Note: overlap is symmetric, d is symmetric.
        # grad_x_i = -2 * lambda_overlap * sum_j overlap_ij * (x_i - x_j) / d_ij
        # Wait, let's check signs.
        # f = (r_i + r_j - d)^2
        # df/dx_i = 2(r_i + r_j - d) * (-1) * (x_i - x_j)/d = -2 * overlap * (x_i - x_j)/d
        # So gradient descent update: x_i -= lr * grad
        # x_i -= lr * (-2 * lambda * overlap * (x_i - x_j)/d)
        # x_i += 2 * lr * lambda * overlap * (x_i - x_j)/d
        # This pushes x_i away from x_j if overlap > 0?
        # If x_i > x_j, x_i - x_j > 0. x_i increases. Correct.
        
        grad_x_overlap = -2 * lambda_overlap * np.sum(overlap * (dx / d_safe), axis=1)
        grad_y_overlap = -2 * lambda_overlap * np.sum(overlap * (dy / d_safe), axis=1)
        grad_r_overlap = 2 * lambda_overlap * np.sum(overlap, axis=1)
        
        # Boundary Penalty
        # f_bound = (r - x)^2 if x < r, else 0.
        # df/dx = -2(r-x) if x < r.
        # df/dr = 2(r-x) if x < r.
        
        bound_x_left = np.maximum(0, r - x)
        bound_x_right = np.maximum(0, r - (1 - x))
        bound_y_bottom = np.maximum(0, r - y)
        bound_y_top = np.maximum(0, r - (1 - y))
        
        grad_x_bound = -2 * lambda_bound * bound_x_left + 2 * lambda_bound * bound_x_right
        grad_y_bound = -2 * lambda_bound * bound_y_bottom + 2 * lambda_bound * bound_y_top
        grad_r_bound = 2 * lambda_bound * (bound_x_left + bound_x_right + bound_y_bottom + bound_y_top)
        
        # Objective: Maximize Sum(r) -> Minimize -Sum(r)
        # grad_r_obj = -1 (since d(-r)/dr = -1)
        grad_r_obj = -np.ones(N)
        
        # Total Gradients
        grad_x = grad_x_overlap + grad_x_bound
        grad_y = grad_y_overlap + grad_y_bound
        grad_r = grad_r_overlap + grad_r_bound + grad_r_obj
        
        # Adaptive learning rate: reduce when gradients are large (avoiding instability)
        grad_mag = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_r**2))
        if grad_mag > 10:
            effective_lr = lr * (10 / grad_mag)
        else:
            effective_lr = lr
        
        # Update
        x = x - effective_lr * grad_x
        y = y - effective_lr * grad_y
        r = r - effective_lr * grad_r
        
        # Clip radii to be positive and reasonable
        r = np.maximum(r, 0.01)
        r = np.minimum(r, 0.4)
        
        # Clip positions to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Slow learning rate decay
        lr *= 0.9995
        
        # Adaptive penalty: increase as we get closer to valid
        if step % 500 == 0 and step > 0:
            # Check if we're making progress
            current_overlap = np.sum(overlap)
            if current_overlap < 0.01:
                lambda_overlap *= 0.9  # Reduce overlap penalty when nearly valid
                lambda_bound *= 0.9
            elif current_overlap > 0.5:
                lambda_overlap *= 1.1  # Increase if lots of overlap
                lambda_bound *= 1.1

    return x, y, r

def _local_search_refine(x, y, r, deadline):
    """
    Refine the configuration by trying to increase radii and moving centers
    to allow for larger radii.
    """
    # First, clamp radii to boundary constraints
    r_max_bound = np.minimum(
        np.minimum(x, 1 - x),
        np.minimum(y, 1 - y)
    )
    r = np.minimum(r, r_max_bound)
    
    # Iteratively repair overlaps by shrinking only problematic circles
    # Then try to expand
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum
        np.fill_diagonal(overlaps, False)
        
        if not np.any(overlaps):
            break
        
        # Find circles involved in overlaps and shrink them
        overlap_count = np.sum(overlaps, axis=1)
        for i in range(N):
            if overlap_count[i] > 0:
                # Shrink this circle based on how many overlaps it has
                r[i] *= (1 - 0.02 * overlap_count[i])
        
        # Re-clamp to boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        
    # Now try to expand radii
    # For each circle, calculate max possible radius given current positions
    # and other circles
    for _ in range(50):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Max radius from boundaries
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                # r_i + r_j <= dist => r_i <= dist - r_j
                max_r = min(max_r, dist - r[j])
            
            max_r = max(0, max_r)
            
            # Try to increase radius slightly
            new_r = min(max_r, r[i] * 1.01)
            if new_r > r[i] + 1e-6:
                r[i] = new_r
                improved = True
        
        if not improved:
            break
            
    # Final check for validity
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    overlaps = d < r_sum
    np.fill_diagonal(overlaps, False)
    
    # Check boundaries
    bound_violations = (r > x) | (r > 1 - x) | (r > y) | (r > 1 - y)
    
    if not np.any(overlaps) and not np.any(bound_violations):
        return x, y, r
    else:
        # If invalid, shrink to make valid
        # This is a fallback
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound * 0.9)
        return x, y, r

def construct():
    """Main function to construct circle packing."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Initial configurations
    configs = [
        _make_initial_grid(),
        _make_initial_random(1),
        _make_initial_random(2),
        _make_initial_random(3),
        _make_initial_random(4),
        _make_initial_clustered(10),
        _make_initial_clustered(11)
    ]
    
    for config in configs:
        if time.time() > deadline:
            break
            
        x = config[:, 0].copy()
        y = config[:, 1].copy()
        r = config[:, 2].copy()
        
        # Gradient Descent
        x, y, r = _gradient_descent(x, y, r, deadline)
        
        # Local Search Refinement
        x, y, r = _local_search_refine(x, y, r, deadline)
        
        # Post-process to ensure validity
        circles = np.column_stack((x, y, r))
        circles = _post_process(circles, deadline)
        
        score = _score(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
            
    if best_circles is None:
        # Fallback
        best_circles = _make_initial_grid()
        best_circles[:, 2] = 0.01
        
    return best_circles
