"""Pack 50 equal circles in a unit square; maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5


def _compute_radius(pts):
    """Compute the common radius for a set of centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # distance to boundary
    dist_boundary = np.min(np.abs(pts), axis=1)
    min_boundary = HALF - np.max(dist_boundary)
    if min_boundary < -1e-12:
        return 0.0
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_boundary, min_pair / 2.0)
    if r < 0:
        return 0.0
    return r


def _soft_violation_obj(pts, beta=50.0):
    """
    Smooth surrogate: maximize the minimum of (boundary distance, pairwise distance/2).
    We minimize the negative of a soft-min approximation.
    Returns (objective_value, gradient).
    """
    if pts.shape[0] != N:
        return 1e10, np.zeros_like(pts)
    
    # Boundary term: for each point i, the distance to the nearest wall is min(|x_i|, |y_i|, 0.5-|x_i|, 0.5-|y_i|)
    # Actually, distance to boundary of [-0.5, 0.5]^2 for point (x,y) is min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    # But simpler: the circle of radius r centered at (x,y) fits if x+r <= 0.5, x-r >= -0.5, etc.
    # So the constraint is x + r <= 0.5 => r <= 0.5 - x, r <= 0.5 + x, r <= 0.5 - y, r <= 0.5 + y
    # The max r for point i is min(0.5-|x_i|, 0.5-|y_i|) ... wait, no.
    # For a point (x, y) in [-0.5, 0.5]^2, the distance to the boundary is min(x+0.5, 0.5-x, y+0.5, 0.5-y).
    # Since x,y in [-0.5, 0.5], x+0.5 >= 0, 0.5-x >= 0.
    # dist_i = min(0.5 - abs(x_i), 0.5 - abs(y_i)) is NOT correct.
    # dist_i = min(x_i + 0.5, 0.5 - x_i, y_i + 0.5, 0.5 - y_i)
    # Let's compute this properly.
    x = pts[:, 0]
    y = pts[:, 1]
    dist_to_walls = np.stack([x + HALF, HALF - x, y + HALF, HALF - y], axis=1)
    boundary_dist = np.min(dist_to_walls, axis=1)  # shape (N,)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0  # shape (N, N)
    
    # Soft min over all constraints: N boundary distances and N*(N-1)/2 pairwise distances
    # We want to maximize r such that r <= boundary_dist_i for all i, and r <= d_ij/2 for all i<j.
    # Equivalent to maximizing soft_min(boundary_dist_i, d_ij/2) over all constraints.
    # Let's collect all constraint values into a flat array.
    # boundary_vals: shape (N,)
    # pair_vals: shape (N*(N-1)/2,)
    pair_idx = np.triu_indices(N, k=1)
    pair_vals = d_half[pair_idx[0], pair_idx[1]]  # shape (N*(N-1)/2,)
    
    all_vals = np.concatenate([boundary_dist, pair_vals])  # shape (N + N*(N-1)/2,)
    
    # Soft min: -log(sum(exp(-beta * v))) / beta
    v_min = np.min(all_vals)
    # For numerical stability
    exp_terms = np.exp(-beta * (all_vals - v_min))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = v_min - log_sum / beta
    
    # Gradient of soft_min_val w.r.t. pts
    # d(soft_min_val)/d(pts_k) = -1/beta * d(log_sum)/d(pts_k)
    # log_sum = log(sum_j exp(-beta * (v_j - v_min)))
    # d(log_sum)/d(pts_k) = sum_j [ -beta * dv_j/d(pts_k) * exp(-beta*(v_j-v_min)) ] / sum_j exp(-beta*(v_j-v_min))
    # So d(soft_min_val)/d(pts_k) = -1/beta * d(log_sum)/d(pts_k) = sum_j [ dv_j/d(pts_k) * exp(-beta*(v_j-v_min)) ] / sum_j exp(-beta*(v_j-v_min))
    # = sum_j [ dv_j/d(pts_k) * w_j ] where w_j = exp(-beta*(v_j-v_min)) / sum(exp(-beta*(v_j-v_min)))
    
    weights = exp_terms / np.sum(exp_terms)  # shape (N + N*(N-1)/2,)
    
    # Now compute gradient
    grad = np.zeros_like(pts)
    
    # Boundary part: first N weights
    w_boundary = weights[:N]  # shape (N,)
    # boundary_dist_i = min(x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i)
    # The soft min over the 4 walls for each point... wait, I used hard min for boundary_dist.
    # That's a problem. Let me use soft min for the 4 walls per point too.
    # Actually, let's just use the hard min for now and handle the subgradient.
    # For the boundary, the active wall determines the gradient.
    # For point i, the active wall is argmin of [x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i]
    active_wall = np.argmin(dist_to_walls, axis=1)  # shape (N,)
    for i in range(N):
        if w_boundary[i] == 0:
            continue
        wall = active_wall[i]
        if wall == 0:  # x_i + 0.5
            grad[i, 0] += w_boundary[i]
        elif wall == 1:  # 0.5 - x_i
            grad[i, 0] -= w_boundary[i]
        elif wall == 2:  # y_i + 0.5
            grad[i, 1] += w_boundary[i]
        elif wall == 3:  # 0.5 - y_i
            grad[i, 1] -= w_boundary[i]
    
    # Pairwise part: remaining weights
    w_pair = weights[N:]  # shape (N*(N-1)/2,)
    for k in range(len(w_pair)):
        if w_pair[k] == 0:
            continue
        i = pair_idx[0][k]
        j = pair_idx[1][k]
        # v_k = d_ij / 2 = 0.5 * sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # d(v_k)/d(x_i) = 0.5 * (x_i - x_j) / d_ij = (x_i - x_j) / (2 * d_ij)
        # But d_ij = 2 * v_k, so d(v_k)/d(x_i) = (x_i - x_j) / (4 * v_k)
        # Wait: d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2), v_k = d_ij/2
        # d(v_k)/d(x_i) = 0.5 * (x_i - x_j) / d_ij = (x_i - x_j) / (2 * d_ij) = (x_i - x_j) / (4 * v_k)
        # Similarly for other components.
        # But if v_k is very small, this blows up. Let's use d_ij directly.
        d_ij = 2 * v_pair if v_pair > 0 else 1e-10
        # Actually, let me just compute it directly from the difference.
        dx = pts[i, 0] - pts[j, 0]
        dy = pts[i, 1] - pts[j, 1]
        d_ij = np.sqrt(dx*dx + dy*dy)
        if d_ij < 1e-10:
            d_ij = 1e-10
        # d(v_k)/d(x_i) = dx / (2 * d_ij)
        # d(v_k)/d(x_j) = -dx / (2 * d_ij)
        grad[i, 0] += w_pair[k] * dx / (2 * d_ij)
        grad[j, 0] -= w_pair[k] * dx / (2 * d_ij)
        grad[i, 1] += w_pair[k] * dy / (2 * d_ij)
        grad[j, 1] -= w_pair[k] * dy / (2 * d_ij)
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val, -grad


def _lbfgs_optimize(pts, time_budget=20.0, start_time=None):
    """Optimize using L-BFGS-B to maximize the soft-min radius surrogate."""
    if start_time is None:
        start_time = time.time()
    
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            raise ValueError("Time budget exceeded")
        pts_2d = x.reshape(N, 2)
        val, grad = _soft_violation_obj(pts_2d, beta=50.0)
        return val, grad
    
    # Clip initial points to be within bounds
    x0 = np.clip(x0.reshape(N, 2), -HALF, HALF).flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if time.time() - start_time > time_budget:
            return pts, _compute_radius(pts)
        
        best_pts = result.x.reshape(N, 2)
        best_pts = np.clip(best_pts, -HALF, HALF)
        best_r = _compute_radius(best_pts)
        return best_pts, best_r
    except Exception:
        return pts, _compute_radius(pts)


def _hex_init(seed=42):
    """Initialize with a hexagonal-like packing."""
    rng = np.random.RandomState(seed)
    # Create a hexagonal grid pattern
    pts = []
    # Hexagonal packing: rows offset by half a spacing
    spacing_x = 1.0 / 7  # ~7 columns
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    row = 0
    while len(pts) < N:
        x_offset = (spacing_x / 2) if (row % 2 == 1) else 0
        col = 0
        while len(pts) < N and col < 10:
            x = -HALF + spacing_x / 2 + x_offset + col * spacing_x
            y = -HALF + spacing_y / 2 + row * spacing_y
            if -HALF + 0.001 <= x <= HALF - 0.001 and -HALF + 0.001 <= y <= HALF - 0.001:
                pts.append([x, y])
            col += 1
        row += 1
    
    pts = np.array(pts[:N])
    # Add small random jitter
    pts += rng.uniform(-0.005, 0.005, size=pts.shape)
    pts = np.clip(pts, -HALF + 0.001, HALF - 0.001)
    return pts


def _grid_init(seed=42):
    """Initialize with a square grid."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += rng.uniform(-0.005, 0.005, size=pts.shape)
    pts = np.clip(pts, -HALF + 0.001, HALF - 0.001)
    return pts


def _simulated_annealing(pts, time_budget=30.0, start_time=None):
    """Simulated annealing to maximize radius."""
    if start_time is None:
        start_time = time.time()
    
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.01
    T_end = 0.0001
    
    rng = np.random.RandomState(42)
    
    iter_count = 0
    no_improve_count = 0
    patience = 20000
    
    while time.time() - start_time < time_budget:
        progress = min(1.0, iter_count / 300000.0)
        T = T_start * (T_end / T_start) ** progress
        
        idx = rng.randint(N)
        step_size = T * 0.5
        delta = rng.normal(0, step_size, size=2)
        
        new_pts = current.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_r = _compute_radius(new_pts)
        
        if new_r >= current_r:
            current = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            accept_prob = np.exp((new_r - current_r) / T)
            if rng.random() < accept_prob:
                current = new_pts
                current_r = new_r
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        iter_count += 1
        
        if no_improve_count > patience:
            current = best.copy()
            current += rng.uniform(-0.02, 0.02, size=current.shape)
            current = np.clip(current, -HALF, HALF)
            current_r = _compute_radius(current)
            no_improve_count = 0
    
    return best, best_r


def _high_precision_polish(pts, time_budget=10.0, start_time=None):
    """Final polish with very fine-tuned local search."""
    if start_time is None:
        start_time = time.time()
    
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    rng = np.random.RandomState(777)
    
    # Try moving each point in small increments
    for step in [0.001, 0.0005, 0.0001, 0.00005]:
        if time.time() - start_time > time_budget:
            break
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            for coord in range(2):
                for sign in [1, -1]:
                    new_pts = current.copy()
                    new_pts[i, coord] += sign * step
                    new_pts = np.clip(new_pts, -HALF, HALF)
                    new_r = _compute_radius(new_pts)
                    if new_r > current_r:
                        current = new_pts
                        current_r = new_r
                        if current_r > best_r:
                            best_r = current_r
                            best = current.copy()
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = [
        _hex_init(seed=1),
        _grid_init(seed=2),
        _hex_init(seed=3),
        _grid_init(seed=4),
    ]
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        # Phase 1: Simulated Annealing
        sa_budget = min(20.0, deadline - elapsed - 10)
        if sa_budget <= 0:
            break
        sa_pts, sa_r = _simulated_annealing(init_pts, time_budget=sa_budget, start_time=start_time)
        
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
        
        # Phase 2: L-BFGS optimization
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        lbfgs_budget = min(15.0, deadline - elapsed - 5)
        if lbfgs_budget <= 0:
            break
        lbfgs_pts, lbfgs_r = _lbfgs_optimize(sa_pts, time_budget=lbfgs_budget, start_time=start_time)
        
        if lbfgs_r > best_r:
            best_r = lbfgs_r
            best_pts = lbfgs_pts.copy()
        
        # Phase 3: High precision polish
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        polish_budget = min(5.0, deadline - elapsed - 2)
        if polish_budget <= 0:
            break
        polish_pts, polish_r = _high_precision_polish(lbfgs_pts, time_budget=polish_budget, start_time=start_time)
        
        if polish_r > best_r:
            best_r = polish_r
            best_pts = polish_pts.copy()
    
    # Final fallback
    if best_pts is None:
        best_pts = _grid_init(seed=0)
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
