"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_value(x, y, r, W):
    """Compute sum of radii if valid, else 0.0. H = 2 - W."""
    H = 2.0 - W
    if W <= 1e-6 or H <= 1e-6:
        return 0.0
    for i in range(N):
        if x[i] < r[i] - 1e-8 or x[i] > W - r[i] + 1e-8:
            return 0.0
        if y[i] < r[i] - 1e-8 or y[i] > H - r[i] + 1e-8:
            return 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - 1e-8:
                return 0.0
    return float(np.sum(r))


def _penalty_objective(params, W):
    """
    Penalty-based objective for L-BFGS-B.
    params: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
    Returns a smooth penalty that is 0 when valid, positive when violated.
    """
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    H = 2.0 - W
    
    if W <= 0.1 or H <= 0.1:
        return 1e6
    
    penalty = 0.0
    penalty_weight = 100.0
    
    # Boundary violations
    for i in range(n):
        if x[i] < r[i]:
            penalty += penalty_weight * (r[i] - x[i])**2
        if x[i] > W - r[i]:
            penalty += penalty_weight * (x[i] - (W - r[i]))**2
        if y[i] < r[i]:
            penalty += penalty_weight * (r[i] - y[i])**2
        if y[i] > H - r[i]:
            penalty += penalty_weight * (y[i] - (H - r[i]))**2
    
    # Overlap violations (smooth)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += penalty_weight * (min_dist - dist)**2
    
    # Negative radius penalty
    for i in range(n):
        if r[i] < 0.005:
            penalty += penalty_weight * (0.005 - r[i])**2
    
    # If no violations, return negative sum of radii (to maximize)
    if penalty < 1e-10:
        return -np.sum(r)
    
    return penalty - 0.1 * np.sum(r)


def _seed_grid(W):
    """Create a grid of circles in a W x (2-W) rectangle."""
    H = 2.0 - W
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2*side) - 1e-4
            cells.append((cx, cy, max(r, 0.005)))
    return np.array(cells[:N])


def _optimize_scipy(W, deadline):
    """Optimize circle positions and radii using L-BFGS-B with penalty."""
    H = 2.0 - W
    initial = _seed_grid(W)
    n = N
    
    x0 = np.concatenate([initial[:, 0], initial[:, 1], initial[:, 2]])
    
    # Bounds: x in [r, W-r], y in [r, H-r], r in [0.005, min(W,H)/2]
    max_r = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.005, W - 0.005))  # x
    for i in range(n):
        bounds.append((0.005, H - 0.005))  # y
    for i in range(n):
        bounds.append((0.005, max_r))  # r
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        return _penalty_objective(params, W)
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                               'maxfun': 5000})
    
    if time.time() > deadline:
        return None, 0.0
    
    params = result.x
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    val = _compute_value(x, y, r, W)
    if val > 0:
        return np.column_stack([x, y, r]), val
    return None, 0.0


def _local_refine(result, W, deadline):
    """Local refinement using small perturbations after scipy optimization."""
    if result is None:
        return None, 0.0
    
    H = 2.0 - W
    best_x = result[:, 0].copy()
    best_y = result[:, 1].copy()
    best_r = result[:, 2].copy()
    best_val = _compute_value(best_x, best_y, best_r, W)
    
    if best_val == 0:
        return None, 0.0
    
    step_size = 0.02
    for it in range(300):
        if time.time() > deadline:
            break
        
        x_new = best_x.copy()
        y_new = best_y.copy()
        r_new = best_r.copy()
        
        idx = np.random.randint(N)
        x_new[idx] += np.random.uniform(-step_size, step_size)
        y_new[idx] += np.random.uniform(-step_size, step_size)
        r_new[idx] = max(0.005, r_new[idx] + np.random.uniform(-0.005, 0.005))
        
        x_new[idx] = np.clip(x_new[idx], r_new[idx], W - r_new[idx])
        y_new[idx] = np.clip(y_new[idx], r_new[idx], H - r_new[idx])
        
        val = _compute_value(x_new, y_new, r_new, W)
        if val > best_val:
            best_x = x_new
            best_y = y_new
            best_r = r_new
            best_val = val
            step_size *= 0.95
        else:
            step_size = min(0.05, step_size * 1.03)
    
    return np.column_stack([best_x, best_y, best_r]), best_val


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    best_result = None
    best_val = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        result, val = _optimize_scipy(W, deadline)
        if result is not None:
            # Local refinement
            refined, ref_val = _local_refine(result, W, deadline)
            if refined is not None and ref_val > best_val:
                best_val = ref_val
                best_result = refined
    
    # If no valid result found, fall back to a simple grid
    if best_result is None:
        initial = _seed_grid(1.0)
        best_result = initial
        best_val = _compute_value(initial[:, 0], initial[:, 1], initial[:, 2], 1.0)
    
    return best_result
# EVOLVE-BLOCK-END
