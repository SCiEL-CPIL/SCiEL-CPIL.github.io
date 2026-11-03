"""Pack 200 equal circles in a unit square using vectorized force-directed + Projected Gradient Descent."""
import numpy as np
import time

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance using vectorized calculation
    # N=200, so 200x200 matrix is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_initial(spacing=0.07, n_rows=None, n_cols=None, jitter=0.0):
    """Generate a hexagonal initial configuration with exactly N points."""
    if n_rows is None or n_cols is None:
        # Estimate rows/cols for N points
        n_cols = int(np.sqrt(N * 2 / np.sqrt(3)))
        n_rows = int(np.ceil(N / n_cols))
    
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    y_start = -0.5 + spacing / 2
    x_start = -0.5 + spacing / 2
    
    for i in range(n_rows):
        shift = (col_w / 2.0) if i % 2 == 1 else 0.0
        for j in range(n_cols):
            x = x_start + j * col_w + shift
            y = y_start + i * row_h
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have too many, randomly select N
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    elif len(pts) < N:
        # Add random points to fill
        while len(pts) < N:
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            pts = np.vstack([pts, [[x, y]]])
    
    if jitter > 0:
        pts += np.random.randn(len(pts), 2) * jitter
    
    # Clip to bounds
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts[:N]


def _soft_min_value_and_grad(pts, beta=50.0):
    """
    Compute soft-min objective and gradient using vectorized operations.
    Objective: -1/beta * log(sum(exp(-beta * c_k)))
    c_k are wall distances and half pairwise distances.
    """
    n = len(pts)
    
    # 1. Wall distances and gradients
    # c_wall: shape (4n, )
    # grad_wall: shape (n, 2)
    
    # x + 0.5, 0.5 - x, y + 0.5, 0.5 - y
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Gradients for wall constraints
    # d/dx (x+0.5) = 1, d/dy = 0
    # d/dx (0.5-x) = -1, d/dy = 0
    # d/dx (y+0.5) = 0, d/dy = 1
    # d/dx (0.5-y) = 0, d/dy = -1
    grad_wall = np.zeros((n, 2))
    grad_wall[:, 0] += 1.0 # from x+0.5
    grad_wall[:, 0] += -1.0 # from 0.5-x -> wait, this is sum of gradients?
    
    # Actually, we need to weight them by the softmax probabilities later.
    # Let's store the raw gradients for each constraint.
    # There are 4 constraints per point.
    # grad_wall[i, 0] is the x-component of the gradient for the 4 constraints combined? No.
    # We need to compute sum_k w_k * grad_k.
    
    # Let's compute weights first, then accumulate.
    
    # 2. Pairwise distances and gradients
    # diff: (n, n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    pair_vals = dists / 2.0
    
    # Combine all constraint values
    all_vals = np.concatenate([wall_vals, pair_vals.flatten()])
    
    # Compute softmax weights
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    soft_min_value = -(max_neg + np.log(sum_exp)) / beta
    
    weights = exp_shifted / sum_exp
    
    # 3. Accumulate Gradient
    grad = np.zeros((n, 2))
    
    # Wall contributions
    w_wall = weights[:4*n]
    # Reshape w_wall to (n, 4)
    w_wall = w_wall.reshape(n, 4)
    
    # Gradients for the 4 wall constraints:
    # 0: x+0.5 -> [1, 0]
    # 1: 0.5-x -> [-1, 0]
    # 2: y+0.5 -> [0, 1]
    # 3: 0.5-y -> [0, -1]
    
    # grad_x = w0 * 1 + w1 * (-1) + w2 * 0 + w3 * 0
    # grad_y = w0 * 0 + w1 * 0 + w2 * 1 + w3 * (-1)
    grad[:, 0] += w_wall[:, 0] - w_wall[:, 1]
    grad[:, 1] += w_wall[:, 2] - w_wall[:, 3]
    
    # Pair contributions
    w_pair = weights[4*n:]
    w_pair = w_pair.reshape(n, n)
    
    # Gradient for pair (i, j) on point i:
    # d/dp_i (||p_i - p_j||/2) = (p_i - p_j) / (2 * ||p_i - p_j||) = diff[i,j] / (2 * dists[i,j])
    
    # We need sum_j w_pair[i,j] * diff[i,j] / (2 * dists[i,j])
    
    # Handle division by zero (diagonal is inf, so 1/inf = 0, but let's be safe)
    inv_dists = np.zeros_like(dists)
    mask = dists > 1e-10
    inv_dists[mask] = 1.0 / dists[mask]
    
    # Factor: w_pair[i,j] * inv_dists[i,j] * 0.5
    factor = w_pair * inv_dists * 0.5
    
    # grad_pair[i] = sum_j factor[i,j] * diff[i,j]
    # diff is (n, n, 2), factor is (n, n)
    # We can use einsum or broadcasting
    # grad_pair = np.einsum('ij,ijk->ik', factor, diff)
    
    # Let's use broadcasting: factor[:, :, np.newaxis] * diff
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    grad += grad_pair
    
    return soft_min_value, grad


def _compute_forces(pts, r_target, repulsion_strength=10.0, wall_strength=10.0):
    """
    Compute vectorized forces for force-directed relaxation.
    """
    n = len(pts)
    forces = np.zeros((n, 2))
    
    # Wall forces
    # Push in if < -0.5 + r, push out if > 0.5 - r
    forces[:, 0] += wall_strength * np.maximum(0, (-0.5 + r_target) - pts[:, 0])
    forces[:, 0] -= wall_strength * np.maximum(0, pts[:, 0] - (0.5 - r_target))
    forces[:, 1] += wall_strength * np.maximum(0, (-0.5 + r_target) - pts[:, 1])
    forces[:, 1] -= wall_strength * np.maximum(0, pts[:, 1] - (0.5 - r_target))
    
    # Pair forces
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Overlap = 2*r_target - dist
    overlap = np.maximum(0, 2 * r_target - dists)
    
    # Force magnitude proportional to overlap / dist^2 (or similar)
    # F = k * overlap * (p_i - p_j) / dist^2 ?
    # Let's use F = k * (p_i - p_j) / dist * overlap
    
    # Factor for vector part: overlap / dist
    inv_dists = np.zeros_like(dists)
    mask = dists > 1e-10
    inv_dists[mask] = 1.0 / dists[mask]
    
    factor = repulsion_strength * overlap * inv_dists
    
    # Force on i from j: factor[i,j] * diff[i,j]
    # Sum over j
    forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return forces


def _force_directed_phase(pts, r_target, max_iters=50, dt=0.01, time_limit=None):
    """
    Force-directed relaxation.
    """
    pts = pts.copy()
    
    for i in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        forces = _compute_forces(pts, r_target)
        pts += dt * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Cool down
        dt *= 0.95
        
    return pts


def _projected_gradient_descent(pts, beta_schedule, max_iters_per_beta=100, lr=0.01, time_limit=None):
    """
    Projected Gradient Descent on soft-min objective.
    """
    pts = pts.copy()
    
    for beta in beta_schedule:
        if time_limit is not None and time.time() > time_limit:
            break
            
        current_val = np.inf
        for i in range(max_iters_per_beta):
            if time_limit is not None and time.time() > time_limit:
                break
                
            val, grad = _soft_min_value_and_grad(pts, beta)
            
            # Check for improvement
            if val < current_val - 1e-12:
                current_val = val
                # Line search or fixed step
                # Simple step
                new_pts = pts - lr * grad
                new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5, 0.5)
                new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5, 0.5)
                
                # Verify improvement in true radius? Or just soft min?
                # Soft min is a proxy. We can check true radius occasionally.
                pts = new_pts
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
                    
    return pts


def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Generate multiple initial configurations (Hex, Random)
    # 2. Force-directed relaxation to untangle
    # 3. Projected Gradient Descent on soft-min
    
    initial_configs = []
    
    # Hexagonal lattices
    spacings = [0.065, 0.07, 0.075, 0.08]
    for spacing in spacings:
        for jitter in [0.0, 0.01]:
            pts = _generate_hex_initial(spacing=spacing, jitter=jitter)
            if len(pts) == N:
                initial_configs.append(pts.copy())
                
    # Random
    initial_configs.append(np.random.uniform(-0.5, 0.5, (N, 2)))
    
    for init_pts in initial_configs:
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
            
        r_init = _get_radius(init_pts)
        
        # Force Directed Phase
        # Target radius slightly larger than current to encourage expansion
        fd_pts = _force_directed_phase(
            init_pts, 
            r_target=r_init * 1.05, 
            max_iters=30, 
            dt=0.01,
            time_limit=_TIME_LIMIT - 5
        )
        r_fd = _get_radius(fd_pts)
        
        # Projected Gradient Descent
        # Use increasing beta for sharper minimum
        pgd_pts = _projected_gradient_descent(
            fd_pts,
            beta_schedule=[10.0, 20.0, 50.0, 100.0],
            max_iters_per_beta=50,
            lr=0.01,
            time_limit=_TIME_LIMIT - 2
        )
        r_pgd = _get_radius(pgd_pts)
        
        # Track best
        candidates = [
            (init_pts, r_init),
            (fd_pts, r_fd),
            (pgd_pts, r_pgd)
        ]
        
        for pts_candidate, r_candidate in candidates:
            if r_candidate > best_r:
                best_r = r_candidate
                best_pts = pts_candidate.copy()
    
    # Final polish on best result
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        final_pts = _projected_gradient_descent(
            best_pts,
            beta_schedule=[50.0, 100.0, 200.0],
            max_iters_per_beta=50,
            lr=0.005,
            time_limit=_TIME_LIMIT - 1
        )
        r_final = _get_radius(final_pts)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts.copy()
            
    # Fallback
    if best_pts is None:
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
