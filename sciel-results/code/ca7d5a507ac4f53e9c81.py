"""Circle packing in unit square: maximize sum of radii for N disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_T0 = time.time()
_DEADLINE = 115.0  # safety margin before the 150 s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _is_valid(circles, tol=1e-7):
    """Check if circles are valid: inside unit square, no overlaps."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Wall check
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(circles):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(circles):
        return float(np.sum(circles[:, 2]))
    return -np.inf


def _make_grid_initial(seed=0):
    rng = np.random.RandomState(seed)
    side = 5
    x = (np.arange(N) % side + 0.5 + rng.randn(N) * 0.08) / side
    y = (np.arange(N) // side + 0.5 + rng.randn(N) * 0.08) / side
    r = np.full(N, 0.5 / side * 0.85)
    x = np.clip(x, 0.02, 0.98)
    y = np.clip(y, 0.02, 0.98)
    return np.column_stack([x, y, r])


def _make_hex_initial(seed=0):
    rng = np.random.RandomState(seed)
    # Hexagonal packing in unit square
    r0 = 0.09
    dx = 2 * r0 * 0.95
    dy = 2 * r0 * 0.95 * np.sqrt(3) / 2
    circles = []
    row = 0
    while len(circles) < N:
        offset = (dx / 2) if (row % 2) else 0
        col = 0
        while len(circles) < N:
            cx = offset + col * dx
            cy = row * dy
            if 0.01 <= cx <= 0.99 and 0.01 <= cy <= 0.99:
                circles.append([cx, cy, r0])
            col += 1
        row += 1
    circles = np.array(circles[:N])
    # Add small jitter
    circles[:, 0] += rng.randn(N) * 0.005
    circles[:, 1] += rng.randn(N) * 0.005
    return circles


def _make_random_initial(seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(N) * 0.7 + 0.15
    y = rng.rand(N) * 0.7 + 0.15
    r = rng.rand(N) * 0.04 + 0.04
    return np.column_stack([x, y, r])


def _make_corner_initial(seed=0):
    """Place circles near corners and edges."""
    rng = np.random.RandomState(seed)
    # 4 corners
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    circles = []
    for cx, cy in corners:
        circles.append([cx + rng.randn() * 0.02, cy + rng.randn() * 0.02, 0.12])
    # Edge midpoints
    edges = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in edges:
        circles.append([cx + rng.randn() * 0.02, cy + rng.randn() * 0.02, 0.08])
    # Fill remaining with random
    while len(circles) < N:
        circles.append([0.2 + rng.rand() * 0.6, 0.2 + rng.rand() * 0.6, 0.05 + rng.rand() * 0.03])
    return np.array(circles[:N])


def _smooth_objective(z, beta):
    """
    Smooth objective using log-sum-exp soft-plus surrogate.
    z: (N*3,) array of [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    beta: sharpness parameter (higher = closer to hard constraints)
    Returns: -sum(r) + penalty
    """
    z = z.reshape(N, 3)
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    
    # Wall violations (smooth)
    def soft_plus(v, b):
        return np.logaddexp(0, b * v) / b
    
    v1 = soft_plus(r - x, beta)
    v2 = soft_plus(r - (1 - x), beta)
    v3 = soft_plus(r - y, beta)
    v4 = soft_plus(r - (1 - y), beta)
    wall_penalty = np.sum(v1) + np.sum(v2) + np.sum(v3) + np.sum(v4)
    
    # Overlap violations (smooth)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    r_sum = r[i_idx] + r[j_idx]
    
    # We want dist >= r_sum, i.e., dist_sq >= r_sum^2
    # Smooth violation: soft_plus(r_sum^2 - dist_sq, beta)
    overlap_viol = soft_plus(r_sum ** 2 - dist_sq, beta)
    overlap_penalty = np.sum(overlap_viol)
    
    penalty = 50.0 * (wall_penalty + overlap_penalty)
    
    return -np.sum(r) + penalty


def _smooth_gradient(z, beta):
    """
    Numerical gradient of the smooth objective.
    """
    eps = 1e-7
    grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += eps
        z_minus = z.copy()
        z_minus[i] -= eps
        grad[i] = (_smooth_objective(z_plus, beta) - _smooth_objective(z_minus, beta)) / (2 * eps)
    return grad


def _scipy_optimize_smooth(z_init, beta, maxiter=200):
    """Run L-BFGS-B optimization on the smooth objective."""
    if _time_left() < 5:
        return z_init, _smooth_objective(z_init, beta)
    
    # Use numerical gradient (analytical is complex due to log-sum-exp)
    result = minimize(
        _smooth_objective,
        z_init,
        args=(beta,),
        method='L-BFGS-B',
        jac=_smooth_gradient,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
    )
    
    return result.x, result.fun


def _relax_and_grow(circles, max_iter=500, lr=0.01):
    """
    Position-based dynamics: relax overlaps and grow radii.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        
        # Relax wall constraints
        for i in range(N):
            # Push away from walls
            if x[i] < r[i]:
                x[i] = r[i] + 0.001
            if 1 - x[i] < r[i]:
                x[i] = 1 - r[i] - 0.001
            if y[i] < r[i]:
                y[i] = r[i] + 0.001
            if 1 - y[i] < r[i]:
                y[i] = 1 - r[i] - 0.001
        
        # Relax overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    # Weighted by inverse radius (smaller circles move more)
                    wi = r[j] / (r[i] + r[j])
                    wj = r[i] / (r[i] + r[j])
                    x[i] -= nx * overlap * wi * 0.5
                    y[i] -= ny * overlap * wi * 0.5
                    x[j] += nx * overlap * wj * 0.5
                    y[j] += ny * overlap * wj * 0.5
                    # Clamp to unit square
                    x[i] = np.clip(x[i], 0.01, 0.99)
                    y[i] = np.clip(y[i], 0.01, 0.99)
                    x[j] = np.clip(x[j], 0.01, 0.99)
                    y[j] = np.clip(y[j], 0.01, 0.99)
        
        # Try to grow radii slightly
        grow_factor = 1.0 + lr * (1.0 - it / max_iter)
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = min(r[i] * grow_factor, max_r * 0.99)
    
    return np.column_stack([x, y, r])


def _high_precision_polish(circles, max_iter=1000):
    """
    High-precision polish: alternate between fine-grained local search and radius growth.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        if _time_left() < 3:
            break
        
        # Fine-grained local search on positions
        step = 0.001 * (1.0 - it / max_iter) + 0.0001
        
        for i in range(N):
            # Try small perturbations
            best_x, best_y, best_score = x[i], y[i], -np.inf
            
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    nx = np.clip(x[i] + dx, r[i], 1 - r[i])
                    ny = np.clip(y[i] + dy, r[i], 1 - r[i])
                    
                    # Quick validity check
                    valid = True
                    for j in range(N):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - x[j]) ** 2 + (ny - y[j]) ** 2)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                        # Wall check
                        if nx < r[i] - 1e-9 or 1 - nx < r[i] - 1e-9:
                            valid = False
                            break
                        if ny < r[i] - 1e-9 or 1 - ny < r[i] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: prefer positions that allow more radius growth
                    max_r_at_pos = min(nx, 1 - nx, ny, 1 - ny)
                    # Also check min distance to other circles
                    min_dist_to_others = np.inf
                    for j in range(N):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - x[j]) ** 2 + (ny - y[j]) ** 2)
                        min_dist_to_others = min(min_dist_to_others, dist)
                    potential_r = min(max_r_at_pos, (min_dist_to_others - r[i]) / 2 + r[i])
                    
                    if potential_r > best_score:
                        best_score = potential_r
                        best_x, best_y = nx, ny
            
            x[i], y[i] = best_x, best_y
        
        # Grow radii where possible
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            # Check against all other circles
            for j in range(N):
                if j == i:
                    continue
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                max_r = min(max_r, (dist - r[j]) / 2)
            max_r = max(1e-6, max_r)
            if max_r > r[i]:
                r[i] = max_r
    
    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_sum_r = 0.0
    best_circles = None
    
    # Generate multiple initializations
    initializations = []
    
    # Grid-based
    for seed in range(4):
        initializations.append(_make_grid_initial(seed))
    
    # Hexagonal
    for seed in range(3):
        initializations.append(_make_hex_initial(seed))
    
    # Random
    for seed in range(3):
        initializations.append(_make_random_initial(seed))
    
    # Corner-based
    for seed in range(2):
        initializations.append(_make_corner_initial(seed))
    
    # Process each initialization
    for init_idx, init_circles in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Step 1: Relax and grow
        relaxed = _relax_and_grow(init_circles, max_iter=300)
        
        # Step 2: Scipy optimization with increasing beta (annealing)
        z = relaxed.flatten()
        for beta in [10, 50, 100, 200, 500]:
            if time.time() > deadline:
                break
            z, _ = _scipy_optimize_smooth(z, beta, maxiter=100)
        
        # Step 3: High-precision polish
        polished = _high_precision_polish(z.reshape(N, 3), max_iter=500)
        
        # Evaluate
        sum_r = np.sum(polished[:, 2])
        if _is_valid(polished):
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_circles = polished.copy()
    
    # Final validation and cleanup
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.02])
                    count += 1
        best_circles = np.array(cells)
    
    # Ensure validity: fix any remaining violations
    x, y, r = best_circles[:, 0].copy(), best_circles[:, 1].copy(), best_circles[:, 2].copy()
    
    # Fix wall violations
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 1e-6)
    
    # Fix overlaps by shrinking radii
    for iteration in range(200):
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2 + 0.0005
                    r[i] = max(1e-6, r[i] - shrink)
                    r[j] = max(1e-6, r[j] - shrink)
                    fixed = True
        if not fixed:
            break
    
    best_circles = np.column_stack([x, y, r])
    
    # Final validity check; if invalid, shrink more aggressively
    if not _is_valid(best_circles, tol=1e-6):
        for i in range(N):
            r[i] *= 0.95
        best_circles = np.column_stack([x, y, r])
    
    return best_circles
