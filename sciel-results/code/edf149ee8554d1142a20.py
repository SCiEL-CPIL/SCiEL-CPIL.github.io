"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 15.0  # seconds


def _penalty(v, W, H):
    """
    v: (n, 3) array [x, y, r]
    W, H: rectangle dimensions
    Returns: (objective_value, penalty_value)
    objective_value = sum(r)
    penalty_value = large penalty for violations
    """
    x = v[:, 0]
    y = v[:, 1]
    r = v[:, 2]
    n = len(r)
    
    obj = np.sum(r)
    
    # Wall violations: circle must be inside [0, W] x [0, H]
    wall_pen = 0.0
    for i in range(n):
        if x[i] - r[i] < 0:
            wall_pen += (x[i] - r[i]) ** 2
        if x[i] + r[i] > W:
            wall_pen += (x[i] + r[i] - W) ** 2
        if y[i] - r[i] < 0:
            wall_pen += (y[i] - r[i]) ** 2
        if y[i] + r[i] > H:
            wall_pen += (y[i] + r[i] - H) ** 2
    
    # Circle-circle overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    return obj, wall_pen + overlap_pen


def _penalized_objective(v, W, H, penalty_weight):
    obj, pen = _penalty(v, W, H)
    return -obj + penalty_weight * pen


def _force_directed_relax(v, W, H, n_iter=200, lr=0.01):
    """
    Force-directed relaxation: push overlapping circles apart,
    pull circles toward center if outside walls.
    """
    v = v.copy()
    n = len(v)
    for it in range(n_iter):
        forces = np.zeros_like(v)
        x = v[:, 0]
        y = v[:, 1]
        r = v[:, 2]
        
        # Wall forces
        for i in range(n):
            if x[i] - r[i] < 0:
                forces[i, 0] += (x[i] - r[i])
            if x[i] + r[i] > W:
                forces[i, 0] -= (x[i] + r[i] - W)
            if y[i] - r[i] < 0:
                forces[i, 1] += (y[i] - r[i])
            if y[i] + r[i] > H:
                forces[i, 1] -= (y[i] + r[i] - H)
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = dx / dist * overlap
                    fy = dy / dist * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        v += lr * forces
        # Keep radii positive
        v[:, 2] = np.maximum(v[:, 2], 1e-4)
    
    return v


def _optimize_for_aspect(W, H, v0, penalty_weight, max_iter, timeout):
    """
    Optimize for a given aspect ratio (W, H).
    v0: initial (n,3) array
    Returns: (best_v, best_obj) or (None, -inf)
    """
    start_time = time.time()
    
    # Force-directed pre-relaxation
    v = _force_directed_relax(v0, W, H, n_iter=100, lr=0.05)
    
    # SLSQP optimization
    def objective(v_flat):
        if time.time() - start_time > timeout:
            return 1e10
        v_arr = v_flat.reshape(-1, 3)
        return _penalized_objective(v_arr, W, H, penalty_weight)
    
    best_v = v
    best_obj = -np.sum(v[:, 2]) + penalty_weight * _penalty(v, W, H)[1]
    
    # Try multiple penalty weights
    for pw in [penalty_weight, penalty_weight * 0.5, penalty_weight * 2]:
        if time.time() - start_time > timeout:
            break
        
        v_curr = v.copy()
        for restart in range(3):
            if time.time() - start_time > timeout:
                break
            
            # Add small perturbation for restarts > 0
            if restart > 0:
                v_curr = v + np.random.randn(*v.shape) * 0.01
                v_curr[:, 2] = np.maximum(v_curr[:, 2], 1e-4)
            
            res = minimize(
                objective,
                v_curr.flatten(),
                method='SLSQP',
                options={'maxiter': max_iter, 'ftol': 1e-10},
                callback=lambda x: None
            )
            
            if time.time() - start_time > timeout:
                break
            
            v_res = res.x.reshape(-1, 3)
            obj_res, pen_res = _penalty(v_res, W, H)
            score_res = -obj_res + pw * pen_res
            
            if score_res < best_obj:
                best_obj = score_res
                best_v = v_res.copy()
            
            # Also try with the actual objective (sum of radii) as a check
            if pen_res < 1e-6:
                if obj_res > np.sum(best_v[:, 2]):
                    best_v = v_res.copy()
                    best_obj = -obj_res
    
    # Final polish with tighter tolerance
    if time.time() - start_time < timeout:
        v_final = best_v.copy()
        for pw in [penalty_weight * 0.1, penalty_weight * 0.01]:
            if time.time() - start_time > timeout:
                break
            res = minimize(
                lambda x: _penalized_objective(x.reshape(-1, 3), W, H, pw),
                v_final.flatten(),
                method='SLSQP',
                options={'maxiter': max_iter // 2, 'ftol': 1e-12}
            )
            v_res = res.x.reshape(-1, 3)
            obj_res, pen_res = _penalty(v_res, W, H)
            if pen_res < 1e-8 and obj_res > np.sum(v_final[:, 2]):
                v_final = v_res.copy()
        best_v = v_final
    
    obj_final, pen_final = _penalty(best_v, W, H)
    return best_v, obj_final


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_obj = -np.inf
    best_v = None
    
    # Grid-based initial layout
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    v_grid = np.array(cells[:N])
    
    # Aspect ratio sweep: W + 2*H = 4 => H = (4 - W) / 2
    # Try various W values
    aspect_ratios = []
    for W in np.linspace(0.8, 3.2, 15):
        H = (4.0 - W) / 2.0
        if H > 0.1:
            aspect_ratios.append((W, H))
    
    # Sort by time budget
    n_aspects = len(aspect_ratios)
    
    for idx, (W, H) in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        
        # Scale initial layout to fit this rectangle
        v0 = v_grid.copy()
        v0[:, 0] = v0[:, 0] * W
        v0[:, 1] = v0[:, 1] * H
        # Scale radii
        scale = min(W, H) / max(1.0, max(W, H))
        v0[:, 2] = v0[:, 2] * scale
        
        # Remaining time for this aspect ratio
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Allocate time: more time for promising aspect ratios
        time_per_aspect = remaining / max(1, n_aspects - idx)
        time_per_aspect = max(2.0, min(time_per_aspect, 30.0))
        
        penalty_weight = 100.0
        max_iter = 200
        
        v_opt, obj = _optimize_for_aspect(W, H, v0, penalty_weight, max_iter, time_per_aspect)
        
        if v_opt is not None and obj > best_obj:
            best_obj = obj
            best_v = v_opt.copy()
    
    # If no valid solution found, fall back to grid
    if best_v is None:
        best_v = v_grid.copy()
        # Scale to perimeter 4 square
        best_v[:, 0] = best_v[:, 0] * 1.0
        best_v[:, 1] = best_v[:, 1] * 1.0
        best_v[:, 2] = best_v[:, 2] * 0.99
    
    # Final validation: ensure perimeter <= 4 and circles are disjoint
    # The optimization should handle this, but let's verify
    x = best_v[:, 0]
    y = best_v[:, 1]
    r = best_v[:, 2]
    
    # Find bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    perimeter = 2 * (W + H)
    
    # If perimeter > 4, scale down
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        best_v[:, 0] = (best_v[:, 0] - x_min) * scale
        best_v[:, 1] = (best_v[:, 1] - y_min) * scale
        best_v[:, 2] = best_v[:, 2] * scale
    
    # Verify no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_v[i, 0] - best_v[j, 0]
            dy = best_v[i, 1] - best_v[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < best_v[i, 2] + best_v[j, 2] - 1e-9:
                # Reduce radii slightly
                overlap = best_v[i, 2] + best_v[j, 2] - dist
                best_v[i, 2] -= overlap / 2
                best_v[j, 2] -= overlap / 2
    
    # Ensure all radii positive
    best_v[:, 2] = np.maximum(best_v[:, 2], 1e-6)
    
    return best_v
