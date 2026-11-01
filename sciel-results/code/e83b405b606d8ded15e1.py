"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START

def _initial_grid(n, w, h):
    """Generate initial positions in a grid pattern within [0,w]x[0,h]."""
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    if rows == 0:
        rows = 1
    if cols == 0:
        cols = 1
    positions = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = (c + 0.5) * w / cols
        y = (r + 0.5) * h / rows
        positions.append([x, y])
    return np.array(positions, dtype=np.float64)

def _compute_radius(positions, w, h, i):
    """Compute the maximum radius for circle i given positions and box dimensions."""
    x, y = positions[i]
    # Distance to box walls
    r_wall = min(x, w - x, y, h - y)
    # Distance to other circles
    min_dist = np.inf
    for j in range(len(positions)):
        if i == j:
            continue
        dx = x - positions[j][0]
        dy = y - positions[j][1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < min_dist:
            min_dist = d
    r_circ = min_dist / 2.0
    return min(r_wall, r_circ)

def _objective_sum_radii(positions, w, h):
    """Objective: sum of radii (to maximize, we negate for minimization)."""
    total = 0.0
    for i in range(len(positions)):
        total += _compute_radius(positions, w, h, i)
    return -total

def _soft_min(x, k=10.0):
    """Soft minimum using log-sum-exp trick for negative values."""
    # soft_min(x) = -soft_max(-x)
    # soft_max(x) = (1/k) * log(sum(exp(k*x_i)))
    neg_x = -x
    max_neg_x = np.max(neg_x)
    if max_neg_x == -np.inf:
        return 0.0
    exp_terms = np.exp(k * (neg_x - max_neg_x))
    log_sum = np.log(np.sum(exp_terms))
    soft_max_neg = max_neg_x + log_sum / k
    return -soft_max_neg

def _smooth_objective(positions, w, h, k=5.0):
    """Smooth surrogate objective using soft-min for radii."""
    total = 0.0
    n = len(positions)
    for i in range(n):
        x, y = positions[i]
        # Distances to walls
        d_walls = np.array([x, w - x, y, h - y])
        r_wall_soft = _soft_min(d_walls, k)
        
        # Distances to other circles
        if n > 1:
            dx = x - positions[:, 0]
            dy = y - positions[:, 1]
            d = np.sqrt(dx*dx + dy*dy)
            d[i] = np.inf  # exclude self
            r_circ_soft = _soft_min(d, k) / 2.0
        else:
            r_circ_soft = np.inf
        
        r_i = min(r_wall_soft, r_circ_soft)
        total += r_i
    return -total

def _smooth_gradient(positions, w, h, k=5.0, eps=1e-6):
    """Numerical gradient of the smooth objective."""
    grad = np.zeros_like(positions)
    base = _smooth_objective(positions, w, h, k)
    for i in range(len(positions)):
        for d in range(2):
            orig = positions[i][d]
            positions[i][d] = orig + eps
            f_plus = _smooth_objective(positions, w, h, k)
            positions[i][d] = orig - eps
            f_minus = _smooth_objective(positions, w, h, k)
            positions[i][d] = orig
            grad[i][d] = (f_plus - f_minus) / (2 * eps)
    return grad

def _refine_positions(positions, w, h, time_limit, max_iter=500):
    """
    Refine positions to maximize sum of radii using smooth surrogate.
    Returns improved positions.
    """
    start_time = time.time()
    n = len(positions)
    pos_flat = positions.flatten().copy()
    
    # Use SLSQP with smooth objective
    def obj(x):
        return _smooth_objective(x.reshape(-1, 2), w, h, k=3.0)
    
    def jac(x):
        return _smooth_gradient(x.reshape(-1, 2), w, h, k=3.0).flatten()
    
    # Bounds: positions within box
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
    
    best_pos = pos_flat.copy()
    best_obj = obj(best_pos)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        result = minimize(
            obj,
            pos_flat,
            jac=jac,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_pos = result.x.copy()
        
        # Check if we've converged
        if not result.success and result.message in ['Maximum number of iterations has been exceeded.',
                                                       'Positive directional derivative for linesearch']:
            # Try a different starting point with small perturbation
            perturbation = np.random.randn(len(pos_flat)) * 0.01 * min(w, h)
            pos_flat = np.clip(pos_flat + perturbation, 0, w if len(pos_flat) % 2 == 0 else h)
            # Actually need to handle 2D bounds properly
            for i in range(n):
                pos_flat[2*i] = np.clip(pos_flat[2*i], 0, w)
                pos_flat[2*i+1] = np.clip(pos_flat[2*i+1], 0, h)
        else:
            pos_flat = result.x.copy()
    
    return best_pos.reshape(-1, 2)

def _greedy_expand(positions, w, h, n_iter=20):
    """
    Greedy binary-search expansion: for each circle, try to expand its radius
    by moving it slightly, if it doesn't violate constraints.
    This is a local polish pass.
    """
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(n_iter):
        for i in range(n):
            # Try small perturbations
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = positions[i][0] + dx
                    new_y = positions[i][1] + dy
                    if new_x < 0 or new_x > w or new_y < 0 or new_y > h:
                        continue
                    
                    # Check if this improves the radius of circle i without hurting others
                    old_r_i = _compute_radius(positions, w, h, i)
                    
                    positions[i][0] = new_x
                    positions[i][1] = new_y
                    new_r_i = _compute_radius(positions, w, h, i)
                    
                    # Check all other circles still have valid radii
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        r_j = _compute_radius(positions, w, h, j)
                        # Just need to ensure no overlap: distance >= r_i + r_j
                        # But our radius is determined by constraints, so if we move i,
                        # r_j might change. Let's just check if the sum of radii improves.
                        pass
                    
                    if new_r_i > old_r_i + 1e-10:
                        # Accept the move
                        pass
                    else:
                        # Revert
                        positions[i][0] -= dx
                        positions[i][1] -= dy
    
    return positions

def _final_expand(positions, w, h):
    """
    Final high-precision polish: for each circle, binary search for the optimal
    position that maximizes its radius given other circles are fixed.
    """
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(5):  # Multiple passes
        for i in range(n):
            # Fix all other circles, optimize position of circle i
            best_x, best_y = positions[i]
            best_r = _compute_radius(positions, w, h, i)
            
            # Grid search around current position
            step = 0.005 * min(w, h)
            for dx in np.arange(-0.05*min(w,h), 0.05*min(w,h)+step, step):
                for dy in np.arange(-0.05*min(w,h), 0.05*min(w,h)+step, step):
                    nx = positions[i][0] + dx
                    ny = positions[i][1] + dy
                    if nx < 0 or nx > w or ny < 0 or ny > h:
                        continue
                    positions[i][0] = nx
                    positions[i][1] = ny
                    r = _compute_radius(positions, w, h, i)
                    if r > best_r:
                        best_r = r
                        best_x, best_y = nx, ny
            
            positions[i][0] = best_x
            positions[i][1] = best_y
    
    return positions

def construct():
    """
    Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii.
    Returns array of shape (n, 3) with [x, y, r] for each circle.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # We'll try different aspect ratios
    
    best_result = None
    best_value = -np.inf
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w/h = ar, w + h = 2
        # w = ar * h, so ar*h + h = 2, h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Generate initial positions
        positions = _initial_grid(N, w, h)
        
        # Refine using smooth surrogate
        if time.time() < deadline:
            positions = _refine_positions(positions, w, h, time_limit=60.0, max_iter=200)
        
        # Final polish
        if time.time() < deadline:
            positions = _final_expand(positions, w, h)
        
        # Compute radii and value
        radii = np.array([_compute_radius(positions, w, h, i) for i in range(N)])
        value = np.sum(radii)
        
        if value > best_value:
            best_value = value
            # Construct result
            result = np.zeros((N, 3))
            for i in range(N):
                result[i] = [positions[i][0], positions[i][1], radii[i]]
            best_result = result
    
    # If no valid result, fall back to simple grid
    if best_result is None:
        # Square case
        w = h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            c = i % side
            r = i // side
            x = (c + 0.5) / side
            y = (r + 0.5) / side
            # Radius limited by grid spacing and walls
            r_val = 0.5 / side - 1e-3
            cells.append([x, y, r_val])
        best_result = np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
