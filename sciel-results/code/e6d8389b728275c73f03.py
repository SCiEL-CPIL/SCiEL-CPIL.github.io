"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _soft_min_score(pts, k=50.0):
    """
    Soft minimum surrogate for optimization.
    Uses log-sum-exp for the pairwise distances and wall distances.
    """
    pts = np.asarray(pts, dtype=float)
    
    # Wall distances
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    
    # Pairwise distances (upper triangle)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    
    # Soft min of all constraints (wall distances and half-pairwise distances)
    # We want to maximize the minimum, so minimize the negative soft-min
    # Soft min of values v_i: -1/k * log(sum(exp(-k*v_i)))
    
    # Wall soft-min
    wall_soft = -np.log(np.sum(np.exp(-k * wall))) / k
    
    # Pairwise: use d/2, so we work with d2/4
    # For large k, this approximates min(d/2)
    # But exp(-k*d/2) where d = sqrt(d2), so use -k*sqrt(d2)/2
    # For numerical stability, subtract max
    d = np.sqrt(d2)
    half_d = d / 2.0
    # Only consider upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    half_d_upper = half_d[triu_mask]
    
    if len(half_d_upper) == 0:
        pair_soft = wall_soft
    else:
        max_val = np.max(half_d_upper)
        pair_soft = (max_val - np.log(np.sum(np.exp(-k * (half_d_upper - max_val))))) / k
    
    return min(wall_soft, pair_soft)

def _neg_soft_min(pts_flat, k=50.0):
    """Objective for optimizer: negative soft-min score."""
    pts = pts_flat.reshape(N, 2)
    return -_soft_min_score(pts, k)

def _soft_min_gradient(pts, k=50.0):
    """
    Compute gradient of the soft-min objective.
    Returns gradient of -soft_min (so we ascend it, or descend the negative).
    """
    pts = np.asarray(pts, dtype=float)
    grad = np.zeros_like(pts)
    
    # Wall contributions
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_weights = np.exp(-k * wall) / np.sum(np.exp(-k * wall))
    
    # Wall gradient: d(wall)/d(pts)
    # For each point, wall = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    for i in range(N):
        x, y = pts[i]
        # Which wall constraint is active?
        dists = [x + 0.5, 0.5 - x, y + 0.5, 0.5 - y]
        # Gradients for each
        grads = [1.0, -1.0, 1.0, -1.0]
        # Weighted gradient for x
        gx = 0.0
        gy = 0.0
        for d_val, g in zip(dists, grads):
            # This is a simplification; the soft min over all 4 walls per point
            pass
        
        # Simpler: compute wall as min of 4 values per point, soft-min over all N*4
        # But let's just use the minimum wall distance approach
        pass
    
    # Let's use a cleaner approach: compute gradient numerically-free
    # Wall: for each point i, wall_i = min(x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i)
    # The soft-min over all wall distances:
    # S_wall = -1/k * log(sum_i exp(-k * wall_i))
    # dS_wall/dx_i = -1/k * (-k * exp(-k*wall_i) * dwall_i/dx_i) / sum_j exp(-k*wall_j)
    #             = exp(-k*wall_i) * dwall_i/dx_i / sum_j exp(-k*wall_j)
    
    # dwall_i/dx_i: if x_i+0.5 is min, d=1; if 0.5-x_i is min, d=-1
    # We'll compute the weighted average based on which constraint is closest
    
    exp_wall = np.exp(-k * wall)
    wall_sum = np.sum(exp_wall)
    
    for i in range(N):
        x, y = pts[i]
        # Determine effective gradient for x
        dx_wall = 0.0
        dy_wall = 0.0
        # Weight proportional to exp(-k * constraint_value)
        c_left = x + 0.5
        c_right = 0.5 - x
        c_bottom = y + 0.5
        c_top = 0.5 - y
        
        w_left = np.exp(-k * c_left)
        w_right = np.exp(-k * c_right)
        w_bottom = np.exp(-k * c_bottom)
        w_top = np.exp(-k * c_top)
        
        w_total = w_left + w_right + w_bottom + w_top
        if w_total > 0:
            dx_wall = (w_left * 1.0 + w_right * (-1.0)) / w_total
            dy_wall = (w_bottom * 1.0 + w_top * (-1.0)) / w_total
        
        # But we need to weight by the overall wall soft-min weight
        # The contribution of point i to the soft-min is exp(-k*wall_i)/sum
        # And dwall_i/dx_i is the gradient
        # So dS_wall/dx_i = exp(-k*wall_i) * dwall_i/dx_i / sum_j exp(-k*wall_j)
        # But dwall_i/dx_i itself is a soft-min over the 4 constraints
        
        grad[i, 0] += exp_wall[i] * dx_wall / wall_sum
        grad[i, 1] += exp_wall[i] * dy_wall / wall_sum
    
    # Pairwise contributions
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    half_d = d / 2.0
    
    # Soft-min over all pairs (upper triangle)
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    exp_pair = np.exp(-k * half_d)
    np.fill_diagonal(exp_pair, 0.0)
    pair_sum = np.sum(exp_pair)
    
    if pair_sum > 0:
        # dS_pair/dx_i = sum_j exp(-k*d_ij/2) * d(d_ij/2)/dx_i / pair_sum
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # d(d_ij/2)/dx_i = (x_i - x_j) / (2 * d_ij)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = exp_pair[i, j] / pair_sum
                if d[i, j] > 1e-10:
                    grad[i, 0] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d[i, j])
                    grad[i, 1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d[i, j])
    
    # We want to maximize soft_min, so gradient of -soft_min is -grad
    # For ascent: pts += lr * grad
    return grad

def _gradient_descent(pts, lr=0.001, iterations=5000, k_start=20.0, k_end=100.0, deadline=None):
    """
    Gradient descent on the soft-min objective with annealing of k.
    """
    current = pts.copy()
    best = pts.copy()
    best_score = _score(best)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal k
        frac = it / max(iterations - 1, 1)
        k = k_start * (k_end / k_start) ** frac
        
        # Compute gradient
        grad = _soft_min_gradient(current, k)
        
        # Update
        current = current + lr * grad
        current = np.clip(current, -0.5, 0.5)
        
        # Evaluate
        score = _score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
    
    return best, best_score

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    Try multiple row configurations.
    """
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5, 5, 5, 10],
        [8, 8, 8, 8, 8, 8, 6, 4],
        [9, 8, 8, 8, 8, 8, 5, 4],
        [10, 9, 8, 8, 8, 7, 6, 4],
        [7, 7, 7, 7, 7, 7, 6, 2],
        [8, 7, 7, 7, 7, 7, 7, 0],
        [7, 8, 7, 8, 7, 8, 7, 0],
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
        
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 != 0) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _momentum_descent(pts, lr=0.0005, momentum=0.9, iterations=10000, 
                       k_start=10.0, k_end=200.0, deadline=None):
    """
    Gradient descent with momentum and adaptive learning rate.
    """
    current = pts.copy()
    velocity = np.zeros_like(current)
    best = pts.copy()
    best_score = _score(best)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal k
        frac = it / max(iterations - 1, 1)
        k = k_start * (k_end / k_start) ** frac
        
        # Adaptive learning rate: decrease over time
        lr_eff = lr * (1.0 - 0.5 * frac)
        
        # Compute gradient
        grad = _soft_min_gradient(current, k)
        
        # Momentum update
        velocity = momentum * velocity + lr_eff * grad
        current = current + velocity
        current = np.clip(current, -0.5, 0.5)
        
        # Evaluate
        score = _score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
        
        # If no improvement for a while, reduce momentum
        if it % 1000 == 0 and it > 0:
            momentum = max(0.5, momentum * 0.95)
    
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_pts = _make_hex_initial()
    best_score = _score(best_pts)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 1: Coarse gradient descent with low k
    if time.time() < deadline - 80.0:
        gd_pts, gd_score = _momentum_descent(
            best_pts, lr=0.0005, momentum=0.9, iterations=5000,
            k_start=5.0, k_end=50.0, deadline=deadline - 60.0
        )
        if gd_score > best_score:
            best_score = gd_score
            best_pts = gd_pts
    
    # Phase 2: Finer gradient descent with higher k
    if time.time() < deadline - 40.0:
        gd_pts, gd_score = _momentum_descent(
            best_pts, lr=0.0002, momentum=0.85, iterations=8000,
            k_start=20.0, k_end=200.0, deadline=deadline - 20.0
        )
        if gd_score > best_score:
            best_score = gd_score
            best_pts = gd_pts
    
    # Phase 3: Very fine tuning
    if time.time() < deadline - 10.0:
        gd_pts, gd_score = _momentum_descent(
            best_pts, lr=0.00005, momentum=0.8, iterations=5000,
            k_start=100.0, k_end=500.0, deadline=deadline - 3.0
        )
        if gd_score > best_score:
            best_score = gd_score
            best_pts = gd_pts
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
