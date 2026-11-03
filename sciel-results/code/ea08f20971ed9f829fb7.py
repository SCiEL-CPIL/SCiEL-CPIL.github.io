import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 142  # Leave margin for 150s limit

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    
    # Wall distances: distance to each of the 4 walls
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape((N, 2))
    
    # Calculate constraints
    # 1. Wall distances
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair terms
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Pair distances
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    pair_vals = pair_dists / 2.0
    
    # Combine all constraint values into a single array C
    C = np.concatenate([wall_vals, pair_vals])
    
    # Step 2: Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Step 3: Compute Gradient
    grad = np.zeros_like(x)
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] * 1.0
    grad_pts[:, 0] += weights[N:2*N] * (-1.0)
    grad_pts[:, 1] += weights[2*N:3*N] * 1.0
    grad_pts[:, 1] += weights[3*N:4*N] * (-1.0)
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    w_pair = weights[4*N:]
    grad_x = w_pair * dx_safe / (2 * dists_safe)
    grad_y = w_pair * dy_safe / (2 * dists_safe)
    
    # Use np.add.at for accumulation
    np.add.at(grad_pts[:, 0], i_idx, grad_x)
    np.add.at(grad_pts[:, 0], j_idx, -grad_x)
    np.add.at(grad_pts[:, 1], i_idx, grad_y)
    np.add.at(grad_pts[:, 1], j_idx, -grad_y)
    
    # Reshape grad_pts back to grad
    grad = grad_pts.flatten()
    
    return obj_val, grad

def _force_directed_phase(pts, time_limit, iterations=None):
    """
    Force-directed repulsion to spread out overlapping circles.
    Uses vectorized inverse-square repulsion between close pairs and boundary repulsion.
    """
    if iterations is None:
        iterations = max(500, int(time_limit * 150))

    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    step = 0.004
    lr_decay = 0.998

    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise force computation
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        threshold = 0.12
        close_mask = (D < threshold) & (D > 1e-10)

        # Compute forces: for each pair (i,j) with D[i,j] < threshold,
        # force on i is proportional to (pts[i] - pts[j]) / D[i,j]^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        inv_d3 = np.where(close_mask, 1.0 / (D**3 + 1e-9), 0.0)
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)

        # Boundary repulsion
        margin = 0.05
        forces[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (margin - (pts[:, 0] + 0.5)) * 3.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.5 - margin, (pts[:, 0] - (0.5 - margin)) * 3.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (margin - (pts[:, 1] + 0.5)) * 3.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.5 - margin, (pts[:, 1] - (0.5 - margin)) * 3.0, 0.0)

        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)

        step *= lr_decay

        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_r, best_pts

def _bottleneck_polish(positions, beta=100.0, max_iter=500, tol=1e-12, deadline=None):
    """
    High-precision polish pass using the soft-min objective with analytical gradients.
    """
    x0 = positions.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            lambda x: _objective_and_grad(x, beta),
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 2)
    except Exception:
        return positions

def _generate_hex_initial(spacing, phase_shift=0.0, margin=0.001):
    """
    Generate a hexagonal lattice packing with given spacing and phase shift.
    Returns N points inside [-0.5, 0.5]^2.
    """
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + row_h / 2 + phase_shift
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + spacing / 2 + shift + (phase_shift if row_idx % 2 == 0 else 0.0)
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]

    if len(pts) < N:
        # Fallback: random
        return np.random.uniform(-0.4, 0.4, (N, 2))

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts

def _generate_jittered_hex(spacing, jitter=0.002, seed=0):
    """
    Generate hex lattice with small random jitter to break symmetry.
    """
    rng = np.random.RandomState(seed)
    pts = _generate_hex_initial(spacing, phase_shift=0.0)
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, -0.499, 0.499)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_radius = -np.inf
    best_positions = None
    
    # Multiple initializations
    np.random.seed(42)
    
    # Hexagonal initializations with different parameters
    hex_inits = [
        _generate_hex_initial(0.072, 0.0),
        _generate_hex_initial(0.072, 0.01),
        _generate_hex_initial(0.074, 0.0),
        _generate_jittered_hex(0.072, 0.002, 0),
        _generate_jittered_hex(0.072, 0.003, 1),
        _generate_jittered_hex(0.074, 0.002, 2),
    ]
    
    # Square grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    hex_inits.append(grid[:N])
    
    # Random initialization
    hex_inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    for init_idx, init_pos in enumerate(hex_inits):
        if time.time() > deadline - 30:
            break
            
        current_pos = init_pos.copy()
        
        # Phase 1: Force-directed relaxation to remove overlaps
        remaining = deadline - time.time()
        if remaining > 15:
            fd_time = min(10, remaining - 15)
            _, current_pos = _force_directed_phase(current_pos, fd_time)
        
        # Phase 2: Soft-min optimization with increasing beta
        for beta in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
                break
            
            x0 = current_pos.flatten()
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, beta),
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                current_pos = result.x.reshape(N, 2)
            except Exception:
                break
        
        # Phase 3: Final high-precision polish
        if time.time() < deadline - 5:
            polished = _bottleneck_polish(current_pos, beta=2000.0, max_iter=300, tol=1e-14, deadline=deadline)
            r_polished = _get_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_positions = polished
        
        # Also check current_pos
        r_current = _get_radius(current_pos)
        if r_current > best_radius:
            best_radius = r_current
            best_positions = current_pos.copy()

    # Fallback
    if best_positions is None:
        best_positions = hex_inits[0]
        best_radius = _get_radius(best_positions)
    
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions
