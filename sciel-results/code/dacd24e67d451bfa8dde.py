"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return False, 0.0
    perim = 2*(W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    value = np.sum(radii)
    return True, value


def _voronoi_inradii(x, y, W, H):
    """Compute the inradius for each point: distance to nearest other point or wall."""
    n = len(x)
    inradii = np.zeros(n)
    for i in range(n):
        # Distance to walls
        d_wall = min(x[i], y[i], W - x[i], H - y[i])
        inradii[i] = d_wall
        # Distance to other points
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            inradii[i] = min(inradii[i], d * 0.5)
    return inradii


def _soft_min(values, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(values)
    return m - math.log(np.sum(np.exp(-beta * (values - m)))) / beta


def _objective(params, W, H, beta=50.0):
    """Objective to minimize: negative sum of soft-min inradii."""
    n = len(params) // 2
    x = params[:n]
    y = params[n:]
    
    # Compute pairwise distances
    inradii = np.zeros(n)
    for i in range(n):
        d_wall = min(x[i], y[i], W - x[i], H - y[i])
        inradii[i] = d_wall
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            inradii[i] = min(inradii[i], d * 0.5)
    
    # Use soft min to get a smooth objective
    total = _soft_min(inradii, beta)
    return -total


def _objective_grad(params, W, H, beta=50.0):
    """Gradient of the objective (numerical)."""
    n = len(params) // 2
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params, W, H, beta)
    for k in range(len(params)):
        params_p = params.copy()
        params_p[k] += eps
        f_p = _objective(params_p, W, H, beta)
        grad[k] = (f_p - f0) / eps
    return grad


def _seed_grid(W, H, cols, rows):
    """Create a grid of circle centers."""
    centers = []
    for i in range(cols):
        for j in range(rows):
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            centers.append((cx, cy))
    # Take first N
    return np.array(centers[:N])


def _seed_hex(W, H):
    """Create a hexagonal packing arrangement."""
    centers = []
    # Hexagonal close packing
    r = W / 10.0  # approximate spacing
    dy = r * math.sqrt(3)
    for row in range(15):
        y = r + row * dy
        if y > H - r:
            break
        offset = 0.0 if row % 2 == 0 else r
        for col in range(12):
            x = r + offset + col * 2 * r
            if x > W - r:
                break
            centers.append((x, y))
    return np.array(centers[:N])


def _optimize_positions(x0, y0, W, H, deadline, beta=50.0):
    """Optimize positions to maximize sum of inradii."""
    n = len(x0)
    params0 = np.concatenate([x0, y0])
    
    # Bounds for positions
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
    
    best_val = -_objective(params0, W, H, beta)
    best_params = params0.copy()
    
    start_time = time.time()
    
    # Use L-BFGS-B with gradient
    try:
        result = minimize(
            _objective,
            params0,
            args=(W, H, beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < -best_val:
            best_val = -result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if time.time() > deadline:
        x = best_params[:n]
        y = best_params[n:]
        inradii = _voronoi_inradii(x, y, W, H)
        return x, y, inradii, np.sum(inradii)
    
    # Try with higher beta for more precise convergence
    if time.time() < deadline - 5:
        beta2 = 200.0
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(W, H, beta2),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < -best_val:
                best_val = -result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    x = best_params[:n]
    y = best_params[n:]
    inradii = _voronoi_inradii(x, y, W, H)
    return x, y, inradii, np.sum(inradii)


def _local_refine(x, y, W, H, deadline):
    """Local refinement using small perturbations."""
    n = len(x)
    best_x = x.copy()
    best_y = y.copy()
    best_inradii = _voronoi_inradii(best_x, best_y, W, H)
    best_val = np.sum(best_inradii)
    
    step = 0.02
    for _ in range(500):
        if time.time() > deadline:
            break
        idx = np.random.randint(n)
        x_new = best_x.copy()
        y_new = best_y.copy()
        x_new[idx] += np.random.uniform(-step, step)
        y_new[idx] += np.random.uniform(-step, step)
        x_new[idx] = np.clip(x_new[idx], 0, W)
        y_new[idx] = np.clip(y_new[idx], 0, H)
        
        inradii = _voronoi_inradii(x_new, y_new, W, H)
        val = np.sum(inradii)
        if val > best_val:
            best_x = x_new
            best_y = y_new
            best_inradii = inradii
            best_val = val
            step *= 0.97
        else:
            step = min(0.05, step * 1.03)
    
    return best_x, best_y, best_inradii, best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_value = 0.0
    
    # Try different aspect ratios and grid configurations
    aspect_ratios = [
        (1.0, 1.0),      # square
        (1.2, 0.8),      # slightly wide
        (0.8, 1.2),      # slightly tall
        (1.5, 0.5),      # wide
        (0.5, 1.5),      # tall
        (1.333, 0.667),  # 2:1
        (0.667, 1.333),  # 1:2
    ]
    
    grid_configs = [
        (5, 5),
        (7, 3),
        (3, 7),
        (6, 4),
        (4, 6),
        (7, 4),
        (4, 7),
    ]
    
    for W, H in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        for cols, rows in grid_configs:
            if time.time() > deadline - 10:
                break
            
            # Grid seed
            centers = _seed_grid(W, H, cols, rows)
            x0 = centers[:, 0]
            y0 = centers[:, 1]
            
            x, y, inradii, val = _optimize_positions(x0, y0, W, H, deadline, beta=50.0)
            
            # Local refine
            if time.time() < deadline - 5:
                x, y, inradii, val = _local_refine(x, y, W, H, deadline)
            
            if val > best_value:
                best_value = val
                best_result = np.column_stack([x, y, inradii])
        
        # Hexagonal seed
        if time.time() > deadline - 10:
            break
        centers = _seed_hex(W, H)
        if len(centers) >= N:
            x0 = centers[:N, 0]
            y0 = centers[:N, 1]
            x, y, inradii, val = _optimize_positions(x0, y0, W, H, deadline, beta=50.0)
            if time.time() < deadline - 5:
                x, y, inradii, val = _local_refine(x, y, W, H, deadline)
            if val > best_value:
                best_value = val
                best_result = np.column_stack([x, y, inradii])
    
    # Final validation
    if best_result is not None:
        radii = best_result[:, 2]
        x = best_result[:, 0]
        y = best_result[:, 1]
        # Determine W, H from the best result
        # We need to track which W, H gave the best result
        # For now, use the bounding box
        W = max(x + radii, 1e-9)
        H = max(y + radii, 1e-9)
        # Adjust to ensure perimeter <= 4
        perim = 2 * (W + H)
        if perim > 4.0:
            scale = 4.0 / perim
            W *= scale
            H *= scale
            x *= scale
            y *= scale
            radii *= scale
            best_result = np.column_stack([x, y, radii])
        
        is_valid, value = _validate_and_compute(radii, x, y, W, H)
        if not is_valid:
            # Fallback to simple grid
            W = 1.0
            H = 1.0
            centers = _seed_grid(W, H, 5, 5)
            x = centers[:, 0]
            y = centers[:, 1]
            r = 0.5 / 5 - 1e-4
            radii = np.full(N, r)
            best_result = np.column_stack([x, y, radii])
    
    if best_result is None:
        # Ultimate fallback
        W = 1.0
        H = 1.0
        centers = _seed_grid(W, H, 5, 5)
        x = centers[:, 0]
        y = centers[:, 1]
        r = 0.5 / 5 - 1e-4
        radii = np.full(N, r)
        best_result = np.column_stack([x, y, radii])
    
    return best_result
# EVOLVE-BLOCK-END
