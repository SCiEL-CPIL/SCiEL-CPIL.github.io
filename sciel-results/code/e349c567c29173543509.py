import numpy as np
import time
from scipy.optimize import minimize, constraints

N = 26

def _is_valid(centers, radii, tol=1e-6):
    """Check if circles are disjoint and inside unit square."""
    # Boundary check
    b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
    if np.any(radii > b + tol):
        return False
    # Overlap check
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    if np.any(r_sum > dist + tol):
        return False
    return True

def _objective(x, centers0, radii0):
    """Objective: -sum(radii) for minimization."""
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary violations
    b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
    b_viol = np.maximum(0, radii - b)
    
    # Overlap violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = np.maximum(0, r_sum - dist)
    
    penalty = np.sum(b_viol**2) + np.sum(overlap**2)
    return -np.sum(radii) + 100.0 * penalty

def _objective_grad(x, centers0, radii0):
    """Gradient of objective."""
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    grad = np.zeros_like(x)
    
    # Boundary
    b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
    b_viol = np.maximum(0, radii - b)
    
    # Overlap
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist_safe = np.where(dist > 1e-8, dist, 1e-8)
    unit = diff / dist_safe[:, :, np.newaxis]
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = np.maximum(0, r_sum - dist)
    
    # Gradient of -sum(radii)
    grad[2*N:] += -1.0
    
    # Gradient of boundary penalty
    # db/dx_i
    x_min = centers[:, 0]
    x_max = 1 - centers[:, 0]
    y_min = centers[:, 1]
    y_max = 1 - centers[:, 1]
    b_vals = np.stack([x_min, x_max, y_min, y_max], axis=1)
    min_idx = np.argmin(b_vals, axis=1)
    
    db_dx = np.zeros(N)
    db_dy = np.zeros(N)
    for i in range(N):
        if min_idx[i] == 0: db_dx[i] = 1.0
        elif min_idx[i] == 1: db_dx[i] = -1.0
        elif min_idx[i] == 2: db_dy[i] = 1.0
        elif min_idx[i] == 3: db_dy[i] = -1.0
    
    grad_b_pen = 2.0 * b_viol
    grad[2*N:] += 100.0 * grad_b_pen
    grad[:N] += 100.0 * grad_b_pen * db_dx
    grad[N:2*N] += 100.0 * grad_b_pen * db_dy
    
    # Gradient of overlap penalty
    # d(overlap^2)/dr_k = 2 * sum_j overlap_kj
    grad_r_overlap = 2.0 * np.sum(overlap, axis=1)
    grad[2*N:] += 100.0 * grad_r_overlap
    
    # d(overlap^2)/d(centers[i]) = 2 * sum_j overlap_ij * (centers[i]-centers[j])/dist
    grad_c_overlap = 2.0 * np.sum(overlap[:, :, np.newaxis] * unit, axis=1)
    grad[:N] += 100.0 * grad_c_overlap[:, 0]
    grad[N:2*N] += 100.0 * grad_c_overlap[:, 1]
    
    return grad

def _make_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            centers.append([(i + 0.5) / side, (j + 0.5) / side])
    centers = np.array(centers[:N])
    # Estimate radius
    spacing = 1.0 / side
    radii = np.full(N, spacing * 0.45)
    return centers, radii

def _make_hexagonal():
    """Hexagonal packing initialization."""
    centers = []
    # Hexagonal lattice
    dx = 0.15
    dy = dx * np.sqrt(3) / 2
    row = 0
    while len(centers) < N:
        offset = 0 if row % 2 == 0 else dx / 2
        col = 0
        while len(centers) < N:
            x = offset + col * dx
            y = row * dy
            if 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9:
                centers.append([x, y])
            col += 1
        row += 1
    centers = np.array(centers[:N])
    radii = np.full(N, dx * 0.45)
    return centers, radii

def _make_corner_heavy():
    """Initialize with circles near corners."""
    centers = []
    # Corners
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    centers.extend(corners)
    # Fill rest randomly
    np.random.seed(42)
    while len(centers) < N:
        centers.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    centers = np.array(centers[:N])
    radii = np.full(N, 0.08)
    return centers, radii

def _make_random(seed):
    """Random initialization."""
    np.random.seed(seed)
    centers = np.random.uniform(0.1, 0.9, (N, 2))
    radii = np.full(N, 0.05)
    return centers, radii

def _shrink_to_fit(centers, radii):
    """Shrink radii until valid."""
    for _ in range(100):
        if _is_valid(centers, radii):
            return centers, radii, True
        radii *= 0.95
    return centers, radii, _is_valid(centers, radii)

def _optimize_lbfgs(centers, radii, deadline):
    """Optimize using L-BFGS-B."""
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.001, 0.5))  # radius
    
    result = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B',
                      bounds=bounds, args=(centers, radii),
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success or result.fun < -np.sum(radii):
        centers_new = result.x[:2*N].reshape(N, 2)
        radii_new = result.x[2*N:]
        if _is_valid(centers_new, radii_new):
            return centers_new, radii_new
    return centers, radii

def _optimize_slsqp(centers, radii, deadline):
    """Optimize using SLSQP."""
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    result = minimize(_objective, x0, jac=_objective_grad, method='SLSQP',
                      bounds=bounds, args=(centers, radii),
                      options={'maxiter': 300, 'ftol': 1e-10})
    
    if result.success or result.fun < -np.sum(radii):
        centers_new = result.x[:2*N].reshape(N, 2)
        radii_new = result.x[2*N:]
        if _is_valid(centers_new, radii_new):
            return centers_new, radii_new
    return centers, radii

def _grow_radii(centers, radii, deadline):
    """Try to grow radii while maintaining validity."""
    best_score = np.sum(radii)
    best_radii = radii.copy()
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Try growing each radius slightly
        improved = False
        for i in range(N):
            new_radii = radii.copy()
            new_radii[i] *= 1.02
            if _is_valid(centers, new_radii):
                radii = new_radii
                improved = True
                if np.sum(radii) > best_score:
                    best_score = np.sum(radii)
                    best_radii = radii.copy()
        if not improved:
            break
    
    return centers, best_radii

def _polish(centers, radii, deadline):
    """Final polish: local optimization to maximize sum of radii."""
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((radii[i] * 0.9, 0.5))
    
    result = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B',
                      bounds=bounds, args=(centers, radii),
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.fun < -np.sum(radii):
        centers_new = result.x[:2*N].reshape(N, 2)
        radii_new = result.x[2*N:]
        if _is_valid(centers_new, radii_new):
            return centers_new, radii_new
    return centers, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_score = -1
    best_config = None
    
    def update_best(centers, radii):
        nonlocal best_score, best_config
        if _is_valid(centers, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_config = np.column_stack([centers, radii])
    
    # Generate initializations
    initializations = []
    initializations.append(_make_grid())
    initializations.append(_make_hexagonal())
    initializations.append(_make_corner_heavy())
    for seed in range(20):
        initializations.append(_make_random(seed))
    
    # For each initialization, try optimization
    for init_idx, (centers, radii) in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        # Shrink to get valid starting point
        centers, radii, valid = _shrink_to_fit(centers, radii)
        if not valid:
            continue
        
        update_best(centers, radii)
        
        # Try L-BFGS-B
        if time.time() < deadline - 10:
            c_pol, r_pol = _optimize_lbfgs(centers, radii, deadline)
            update_best(c_pol, r_pol)
        
        # Try SLSQP
        if time.time() < deadline - 8:
            c_slq, r_slq = _optimize_slsqp(centers, radii, deadline)
            update_best(c_slq, r_slq)
        
        # Grow radii
        if time.time() < deadline - 5:
            c_grow, r_grow = _grow_radii(centers, radii, deadline)
            update_best(c_grow, r_grow)
        
        # Final polish
        if time.time() < deadline - 3:
            c_fin, r_fin = _polish(centers, radii, deadline)
            update_best(c_fin, r_fin)
    
    # If no valid config found, return trivial one
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
    
    return best_config
