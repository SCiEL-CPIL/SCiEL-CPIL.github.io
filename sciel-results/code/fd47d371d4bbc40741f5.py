import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin

def _time_ok():
    return time.time() < DEADLINE

def _generate_initial(W, H, layout_type='grid'):
    """Generate initial positions and radii."""
    if layout_type == 'grid':
        # Use a 5x5 grid for 21 circles
        kx, ky = 5, 5
        xs = []
        ys = []
        for j in range(ky):
            for i in range(kx):
                if len(xs) >= N:
                    break
                x = (i + 0.5) * W / kx
                y = (j + 0.5) * H / ky
                xs.append(x)
                ys.append(y)
        while len(xs) < N:
            xs.append(W/2)
            ys.append(H/2)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        # Start with very small radii to ensure no initial overlap
        r = np.ones(N) * min(W, H) / 20.0
        return np.column_stack([xs, ys, r])
    else:
        # Random initial positions
        xs = np.random.rand(N) * W
        ys = np.random.rand(N) * H
        r = np.ones(N) * min(W, H) / 20.0
        return np.column_stack([xs, ys, r])

def _penalized_objective(x, W, H, penalty_weight):
    """
    Compute the penalized objective and its gradient.
    Objective: -sum(r) + penalty_weight * (sum of squared violations)
    Violations:
    1. Boundary: max(0, r - cx)^2, max(0, r - (W-cx))^2, etc.
    2. Overlap: max(0, (r_i + r_j)^2 - dist^2)^2
    """
    cx = x[0:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[2*N:3*N] = -1.0
    
    # Boundary penalties
    # Left: cx - r < 0 => violation = r - cx
    viol_left = np.maximum(0, r - cx)
    obj += penalty_weight * np.sum(viol_left**2)
    grad[0:N] -= penalty_weight * 2 * viol_left * 1.0
    grad[2*N:3*N] -= penalty_weight * 2 * viol_left * (-1.0)
    
    # Right: W - cx - r < 0 => violation = r - (W - cx) = cx + r - W
    viol_right = np.maximum(0, cx + r - W)
    obj += penalty_weight * np.sum(viol_right**2)
    grad[0:N] -= penalty_weight * 2 * viol_right * 1.0
    grad[2*N:3*N] -= penalty_weight * 2 * viol_right * 1.0
    
    # Bottom: cy - r < 0 => violation = r - cy
    viol_bottom = np.maximum(0, r - cy)
    obj += penalty_weight * np.sum(viol_bottom**2)
    grad[N:2*N] -= penalty_weight * 2 * viol_bottom * 1.0
    grad[2*N:3*N] -= penalty_weight * 2 * viol_bottom * (-1.0)
    
    # Top: H - cy - r < 0 => violation = cy + r - H
    viol_top = np.maximum(0, cy + r - H)
    obj += penalty_weight * np.sum(viol_top**2)
    grad[N:2*N] -= penalty_weight * 2 * viol_top * 1.0
    grad[2*N:3*N] -= penalty_weight * 2 * viol_top * 1.0
    
    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist2 = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            sum_r2 = sum_r * sum_r
            
            if dist2 < sum_r2:
                v = sum_r2 - dist2
                obj += penalty_weight * v**2
                
                g_val = penalty_weight * 2 * v
                grad[i] -= g_val * (-2 * dx)
                grad[N+i] -= g_val * (-2 * dy)
                grad[2*N+i] -= g_val * (2 * sum_r)
                
                grad[j] -= g_val * (2 * dx)
                grad[N+j] -= g_val * (2 * dy)
                grad[2*N+j] -= g_val * (2 * sum_r)
                
    return obj, grad

def _optimize_single(W, H, x0, penalty_weight=50.0, max_iter=500):
    """
    Use L-BFGS-B to minimize the penalized objective.
    """
    if not _time_ok():
        return x0, 0.0
    
    # Flatten the initial solution
    x_flat = x0.flatten().copy()
    
    # Define bounds: positions in [0, W] and [0, H], radii in [0, min(W,H)/2]
    max_r = min(W, H) / 2.0
    bounds = [(0, W)] * N + [(0, H)] * N + [(0, max_r)] * N
    
    # Use L-BFGS-B with analytical gradient
    result = minimize(
        _penalized_objective,
        x_flat,
        args=(W, H, penalty_weight),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success and _time_ok():
        x_opt = result.x.reshape(N, 3)
        # Compute score as sum of radii (penalty should be ~0 for valid solution)
        score = np.sum(x_opt[:, 2])
        return x_opt, score
    else:
        return x0, np.sum(x0[:, 2])

def _validate_and_fix(x, W, H):
    """
    Ensure the solution is strictly valid.
    Returns (valid, x_fixed, score)
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    
    # Clip positions
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.clip(r, 0, min(W, H)/2)
    
    # Check boundaries
    for i in range(N):
        min_dist = min(cx[i], W - cx[i], cy[i], H - cy[i])
        if r[i] > min_dist:
            r[i] = min_dist * 0.999
    
    # Check overlaps - iterative fix
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    # Shrink both proportionally
                    scale = dist / sum_r * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break
            
    # Final check
    valid = True
    for i in range(N):
        if r[i] < 0:
            valid = False
            break
        if cx[i] < r[i] - 1e-9 or cx[i] > W - r[i] + 1e-9:
            valid = False
            break
        if cy[i] < r[i] - 1e-9 or cy[i] > H - r[i] + 1e-9:
            valid = False
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist2 = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist2 < sum_r*sum_r - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    score = np.sum(r)
    if valid:
        return True, np.column_stack([cx, cy, r]), score
    else:
        return False, None, 0.0

def _polish_solution(x, W, H, penalty_weight=100.0, max_iter=200):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    if not _time_ok():
        return x, 0.0
    
    x_flat = x.flatten().copy()
    max_r = min(W, H) / 2.0
    bounds = [(0, W)] * N + [(0, H)] * N + [(0, max_r)] * N
    
    result = minimize(
        _penalized_objective,
        x_flat,
        args=(W, H, penalty_weight),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    
    if result.success:
        x_opt = result.x.reshape(N, 3)
        score = np.sum(x_opt[:, 2])
        return x_opt, score
    else:
        return x, np.sum(x[:, 2])

def construct():
    best_score = 0.0
    best_out = None
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33]
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        
        # Try multiple initial layouts
        layouts = ['grid', 'random']
        
        for layout in layouts:
            if not _time_ok():
                break
                
            # Try a few random seeds for 'random' layout
            num_starts = 3 if layout == 'random' else 1
            
            for k in range(num_starts):
                if not _time_ok():
                    break
                    
                x0 = _generate_initial(W, H, layout)
                
                # Run optimization
                x_opt, _ = _optimize_single(W, H, x0)
                
                # Validate and fix
                valid, x_fixed, score = _validate_and_fix(x_opt, W, H)
                
                if valid and score > best_score:
                    # Try to polish the solution
                    if _time_ok():
                        x_polished, _ = _polish_solution(x_fixed, W, H)
                        valid_p, x_fixed_p, score_p = _validate_and_fix(x_polished, W, H)
                        if valid_p and score_p > score:
                            score = score_p
                            x_fixed = x_fixed_p
                    
                    best_score = score
                    best_out = x_fixed
                    
    # Fallback
    if best_out is None:
        W, H = 1.0, 1.0
        g = _generate_initial(W, H, 'grid')
        # Force validity for fallback
        _, g_fixed, _ = _validate_and_fix(g, W, H)
        if g_fixed is not None:
            best_out = g_fixed
        else:
            # Last resort: tiny circles
            best_out = np.column_stack([
                np.random.rand(N) * 0.9 + 0.05,
                np.random.rand(N) * 0.9 + 0.05,
                np.ones(N) * 0.01
            ])
            
    return best_out
