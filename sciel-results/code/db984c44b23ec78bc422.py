import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s limit

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

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    # Handle numerical stability
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

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
    # C[0:N] = x_0 + 0.5 -> grad is [1, 0]
    # C[N:2*N] = 0.5 - x_0 -> grad is [-1, 0]
    # C[2*N:3*N] = x_1 + 0.5 -> grad is [0, 1]
    # C[3*N:4*N] = 0.5 - x_1 -> grad is [0, -1]
    
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
    
    # Gradients for each pair term
    # d(dist/2)/dx_i = dx / (2 * dist)
    # d(dist/2)/dx_j = -dx / (2 * dist)
    # Similarly for y.
    
    # We need to accumulate these into grad_pts.
    # Since we have vectorized operations, we can use np.add.at
    # But np.add.at is slow. Let's try a loop if N is small enough, or vectorized addition.
    # For N=200, num_pairs is ~20k. A loop in Python is too slow.
    # Let's use np.add.at for the accumulation.
    
    # grad_pts[i_idx, 0] += weights[4*N:] * dx_safe / (2 * dists_safe)
    # grad_pts[j_idx, 0] -= weights[4*N:] * dx_safe / (2 * dists_safe)
    # grad_pts[i_idx, 1] += weights[4*N:] * dy_safe / (2 * dists_safe)
    # grad_pts[j_idx, 1] -= weights[4*N:] * dy_safe / (2 * dists_safe)
    
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

def _bottleneck_polish(positions, beta=100.0, max_iter=500, tol=1e-12):
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

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Hexagonal grid initialization
    def hexagonal_init(n, r_target=0.036):
        s = 2 * r_target * 1.001
        points = []
        for i in range(-15, 15):
            for j in range(-15, 15):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
               (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        if len(points) > n:
            dists_to_center = np.sum(points**2, axis=1)
            dist_x = np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5)
            dist_y = np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
            dist_b = np.minimum(dist_x, dist_y)
            score = dists_to_center - 0.1 * dist_b
            idx = np.argsort(score)[:n]
            points = points[idx]
        elif len(points) < n:
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])
            
        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    best_radius = -np.inf
    best_positions = None
    
    # Multiple initializations
    np.random.seed(42)
    inits = [
        hexagonal_init(N, 0.036),
        hexagonal_init(N, 0.035),
        square_grid_init(N),
        np.random.uniform(-0.45, 0.45, size=(N, 2))
    ]
    
    for init_pos in inits:
        if time.time() > deadline - 20:
            break
            
        current_pos = init_pos.copy()
        
        # Start with a conservative radius and inflate
        r = 0.025
        step = 0.002
        
        while r < 0.042:
            if time.time() > deadline - 15:
                break
                
            # Optimize with soft-min objective
            x0 = current_pos.flatten()
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, 50.0),
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                )
                current_pos = result.x.reshape(N, 2)
            except Exception:
                break
            
            # Check feasibility
            actual_r = _get_radius(current_pos)
            
            if actual_r >= r:
                r += step
            else:
                # Binary search for max feasible r
                r_low = r - step
                r_high = r
                
                for _ in range(12):
                    if time.time() > deadline - 10:
                        break
                    r_mid = (r_low + r_high) / 2
                    
                    try:
                        res_mid = minimize(
                            lambda x: _objective_and_grad(x, 50.0),
                            current_pos.flatten(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                        )
                        current_pos = res_mid.x.reshape(N, 2)
                        mid_r = _get_radius(current_pos)
                        if mid_r >= r_mid:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                    except Exception:
                        break
                
                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()
                
                break
        
        if r >= 0.042:
            actual_r = _get_radius(current_pos)
            if actual_r > best_radius:
                best_radius = actual_r
                best_positions = current_pos.copy()

    # Fallback
    if best_positions is None:
        best_positions = square_grid_init(N)
        best_radius = _get_radius(best_positions)
    
    # Final high-precision polish
    if time.time() < deadline - 15:
        # Use bottleneck polish
        polished = _bottleneck_polish(best_positions, beta=100.0, max_iter=500, tol=1e-12)
        r_polished = _get_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished
    
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions
