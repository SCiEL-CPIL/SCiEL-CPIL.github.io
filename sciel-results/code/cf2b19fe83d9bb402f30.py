"""Circle packing in perimeter-4 rectangle: Direct Optimization with Simulated Annealing and Gradient Descent."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _check_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Vectorized overlap check
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    # Check upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol)**2):
        return False
    return True

def _objective(params, W):
    """
    Objective function for scipy.optimize.minimize.
    params: array of shape (3*N,) containing [x1, y1, r1, x2, y2, r2, ...]
    W: width of the rectangle. H = 2 - W.
    Returns: -sum(r) + penalty terms.
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 1e6
    
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    # Ensure radii are positive
    r = np.maximum(r, 1e-6)
    
    # Boundary penalties
    # x should be in [r, W-r]
    # y should be in [r, H-r]
    penalty_x = np.sum(np.maximum(0, r - cx)**2) + np.sum(np.maximum(0, cx - (W - r))**2)
    penalty_y = np.sum(np.maximum(0, r - cy)**2) + np.sum(np.maximum(0, cy - (H - r))**2)
    
    # Overlap penalties
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    d = np.sqrt(np.maximum(d2, 1e-12))
    overlaps = np.maximum(0, rs - d)
    penalty_overlap = np.sum(overlaps[mask]**2)
    
    # Objective: Minimize -sum(r) + lambda * penalties
    # We want to maximize sum(r), so minimize -sum(r)
    # Large penalties for violations
    return -np.sum(r) + 10.0 * (penalty_x + penalty_y + penalty_overlap)

def _optimize_local(cx, cy, r, W, time_limit=5.0):
    """
    Run local optimization using L-BFGS-B.
    """
    start_time = time.time()
    params = np.concatenate([cx, cy, r])
    
    # Bounds for x, y, r
    # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
    # We use soft constraints in objective, but hard bounds help optimizer.
    # However, W and H change. We fix W for this local step.
    H = 2.0 - W
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6)) # x_i
        bounds.append((1e-6, H - 1e-6)) # y_i
        bounds.append((1e-6, min(W, H)/2.0)) # r_i
    
    # Use L-BFGS-B
    try:
        res = minimize(_objective, params, args=(W,), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9, 'maxls': 20})
        
        if res.success or res.fun < _objective(params, W):
            new_params = res.x
            new_cx = new_params[0::3]
            new_cy = new_params[1::3]
            new_r = new_params[2::3]
            
            # Post-process: ensure validity and adjust if needed
            # The optimizer might leave slight violations. We can try to project or shrink.
            # For now, return the result. The caller will check validity.
            return new_cx, new_cy, new_r, W
    except Exception:
        pass
        
    return cx, cy, r, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter 4 -> W + H = 2.
    # W in [0.5, 1.5]
    
    # Strategy:
    # 1. Generate random initial configurations for various W.
    # 2. Use local optimization (L-BFGS-B) to improve them.
    # 3. Keep the best valid configuration.
    
    num_restarts = 20
    W_candidates = np.linspace(0.6, 1.4, 10)
    
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        W = random.choice(W_candidates)
        H = 2.0 - W
        
        # Random initialization
        # Start with small radii to avoid huge initial penalties
        r_init = np.full(N, 0.1)
        cx_init = np.random.uniform(0.1, W - 0.1, N)
        cy_init = np.random.uniform(0.1, H - 0.1, N)
        
        # Run local optimization
        res_cx, res_cy, res_r, res_W = _optimize_local(
            cx_init, cy_init, r_init, W, 
            time_limit=5.0
        )
        
        # Check validity
        if _check_valid(res_cx, res_cy, res_r, res_W, 2.0 - res_W):
            val = np.sum(res_r)
            if val > best_val:
                best_val = val
                best_result = (res_cx, res_cy, res_r, res_W, 2.0 - res_W)
                
    # If no valid result found (unlikely), fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        cell_w = W / side
        cell_h = H / side
        for i in range(N):
            row = i // side
            col = i % side
            cx[i] = (col + 0.5) * cell_w
            cy[i] = (row + 0.5) * cell_h
            r[i] = min(cell_w, cell_h) * 0.4
        return np.column_stack((cx, cy, r))
        
    cx, cy, r, W, H = best_result
    
    # Final Polish: Try to adjust W and re-optimize
    # We can try small perturbations of W and re-run the optimizer
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        W_trial = W + random.uniform(-0.05, 0.05)
        W_trial = np.clip(W_trial, 0.5, 1.5)
        H_trial = 2.0 - W_trial
        
        # Use current positions as initial guess for new W
        # Scale positions? No, just use them.
        # But we need to ensure they are within new bounds.
        cx_trial = np.clip(cx, 0.01, W_trial - 0.01)
        cy_trial = np.clip(cy, 0.01, H_trial - 0.01)
        
        res_cx, res_cy, res_r, res_W = _optimize_local(
            cx_trial, cy_trial, r, W_trial,
            time_limit=2.0
        )
        
        if _check_valid(res_cx, res_cy, res_r, res_W, 2.0 - res_W):
            val = np.sum(res_r)
            if val > best_val:
                best_val = val
                best_result = (res_cx, res_cy, res_r, res_W, 2.0 - res_W)
                
    cx, cy, r, W, H = best_result
    
    # Final safety check
    if not _check_valid(cx, cy, r, W, H):
        # Shrink until valid
        scale = 1.0
        while not _check_valid(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _check_valid(cx, cy, r, W, H):
             # Extreme fallback
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
