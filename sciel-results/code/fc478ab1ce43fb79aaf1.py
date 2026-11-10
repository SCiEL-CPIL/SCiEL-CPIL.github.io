"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_of_placement(centers):
    """Compute the maximum common radius for a placement."""
    # Distance to walls
    walls = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(walls)
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _log_objective(x):
    """Maximize sum of log distances (geometric mean proxy).
    Negative because we minimize."""
    centers = x.reshape(-1, 2)
    # Pairwise distances
    d = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1))
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    # Wall distances
    walls = np.minimum(0.5 - np.abs(centers), axis=1)
    
    # Log-sum-exp for stability: sum(log(d_ij)) over i<j
    # Use only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu]
    log_pairs = np.log(pair_dists + 1e-12)
    
    log_walls = np.log(walls + 1e-12)
    
    total_log = np.sum(log_pairs) + np.sum(log_walls)
    return -total_log


def _log_objective_grad(x):
    """Gradient of negative log-sum objective."""
    centers = x.reshape(-1, 2)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Avoid division by zero
    d_safe = np.where(d > 1e-12, d, 1e-12)
    
    # Gradient contribution from pairs: -sum_j (1/d_ij) * (c_i - c_j)/d_ij
    # For pair (i,j): grad_i += -(c_i - c_j)/d_ij^2, grad_j += (c_i - c_j)/d_ij^2
    inv_d2 = 1.0 / (d_safe ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    grad_pairs = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Gradient contribution from walls: -1/wall * grad(wall)
    # wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This is nonsmooth; use subgradient or smooth approximation
    # Smooth: use 0.5 - 0.5*(|x|+|y|) as surrogate? No, let's use the actual min with subgradient
    # Better: use a smooth wall distance: 0.5 - sqrt(x^2 + y^2)? No, that's not right for square.
    # Use: wall = min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    # Subgradient: pick the active constraint
    
    walls_x = 0.5 - centers[:, 0]
    walls_x_neg = 0.5 + centers[:, 0]
    walls_y = 0.5 - centers[:, 1]
    walls_y_neg = 0.5 + centers[:, 1]
    
    wall_vals = np.stack([walls_x, walls_x_neg, walls_y, walls_y_neg], axis=1)
    min_idx = np.argmin(wall_vals, axis=1)
    
    grad_walls = np.zeros_like(centers)
    for i in range(N):
        idx = min_idx[i]
        w = wall_vals[i, idx]
        if w < 1e-12:
            w = 1e-12
        if idx == 0:  # 0.5 - x
            grad_walls[i, 0] = -1.0 / w
        elif idx == 1:  # 0.5 + x
            grad_walls[i, 0] = 1.0 / w
        elif idx == 2:  # 0.5 - y
            grad_walls[i, 1] = -1.0 / w
        else:  # 0.5 + y
            grad_walls[i, 1] = 1.0 / w
    
    return -(grad_pairs + grad_walls)


def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization."""
    centers = []
    spacing = 1.0 / np.sqrt(n)
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    row = 0
    while len(centers) < n:
        offset = (row % 2) * dx / 2
        col = 0
        while len(centers) < n:
            x = -0.5 + dx/2 + col * dx + offset
            y = -0.5 + dy/2 + row * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                centers.append([x, y])
            col += 1
        row += 1
    
    return np.array(centers[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initialize with hexagonal lattice
    centers = _hex_lattice_init(N)
    x0 = centers.flatten()
    
    best_x = x0.copy()
    best_radius = _radius_of_placement(centers)
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            _log_objective,
            x0,
            jac=_log_objective_grad,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            result_centers = result.x.reshape(-1, 2)
            result_radius = _radius_of_placement(result_centers)
            if result_radius > best_radius:
                best_x = result.x.copy()
                best_radius = result_radius
    except Exception:
        pass
    
    # Check time and do refinement if available
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Polish with more iterations from best point
        try:
            result2 = minimize(
                _log_objective,
                best_x,
                jac=_log_objective_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                result_centers = result2.x.reshape(-1, 2)
                result_radius = _radius_of_placement(result_centers)
                if result_radius > best_radius:
                    best_x = result2.x.copy()
                    best_radius = result_radius
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)
# EVOLVE-BLOCK-END
