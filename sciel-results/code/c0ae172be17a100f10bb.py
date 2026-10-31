"""
Combined approach: Vectorized penalty-based NLP with force-directed pre-conditioning.
Task: Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140.0  # Safety margin under 150s limit

def _vectorized_penalty_objective(v, W, H, penalty_weight=100.0):
    """
    Vectorized objective function.
    v: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
    W, H: dimensions of the rectangle
    Returns: -sum(radii) + penalty terms
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(r)
    
    # 1. Overlap Penalty
    # Compute pairwise distances and radius sums
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    
    r_sum = r[:, None] + r[None, :]
    
    # Only consider upper triangle (i < j) to avoid double counting and self-overlap
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Overlap amount = r_sum - distance
    # We use a quadratic penalty on the violation
    # violation = max(0, r_sum - sqrt(d_sq))
    # To keep it smooth and fast, we can approximate or just use the squared violation
    # Let's use the direct formulation for accuracy, vectorized
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_viol = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap_viol * overlap_viol)
    
    # 2. Boundary Penalty
    # Left: x - r >= 0  -> viol = max(0, r - x)
    # Right: x + r <= W -> viol = max(0, x + r - W)
    # Bottom: y - r >= 0
    # Top: y + r <= H
    
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - W, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - H, 0.0)
    
    bound_viol_sq = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                    np.sum(bottom_viol**2) + np.sum(top_viol**2)
    obj += penalty_weight * bound_viol_sq
    
    return obj

def _force_relax(v, W, H, max_iter=100):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Modifies v in-place (or returns a copy).
    """
    v = v.copy()
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    dt = 0.01
    for _ in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            if x[i] > W - r[i]:
                fx[i] -= (x[i] - (W - r[i])) * 10.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            if y[i] > H - r[i]:
                fy[i] -= (y[i] - (H - r[i])) * 10.0
        
        # Pairwise repulsion
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                
                if d_sq < min_d * min_d:
                    d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-6
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    # Force is proportional to overlap
                    f = overlap * 50.0
                    if d > 1e-12:
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f
                    else:
                        # Random kick if exactly overlapping
                        fx[i] += np.random.uniform(-1, 1) * f
                        fy[i] += np.random.uniform(-1, 1) * f
                        fx[j] -= np.random.uniform(-1, 1) * f
                        fy[j] -= np.random.uniform(-1, 1) * f
        
        if max_overlap < 1e-6:
            break
            
        # Adaptive step size
        force_mag = np.max(np.sqrt(fx * fx + fy * fy))
        if force_mag > 1.0:
            dt = max(0.001, dt * 0.5)
        else:
            dt = min(0.05, dt * 1.1)
            
        x += dt * fx
        y += dt * fy
        
        # Hard clamp to prevent explosion
        x = np.clip(x, 0, W)
        y = np.clip(y, 0, H)
        
        v[0::3] = x
        v[1::3] = y
        
    return v

def _validate_and_compute(W, H, v):
    """
    Validate the solution and compute the sum of radii.
    Returns (v, value) if valid, else (None, 0.0).
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    if np.any(r < 1e-8):
        return None, 0.0
        
    # Check boundaries with strict tolerance
    tol = 1e-5
    if np.any(x - r < -tol) or np.any(x + r > W + tol) or \
       np.any(y - r < -tol) or np.any(y + r > H + tol):
        return None, 0.0
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx * dx + dy * dy
            min_d = r[i] + r[j]
            if d_sq < min_d * min_d - 1e-6:
                return None, 0.0
                
    return v, np.sum(r)

def _generate_grid_guess(W, H):
    """
    Generate a grid-based initial guess for 21 circles.
    """
    # 21 circles: try 3x7, 4x5 (20+1), etc.
    # Let's use a simple 5x5 grid and place 21 in the first 21 spots, or 3x7
    # 3 rows, 7 cols is a good aspect ratio for many rectangles.
    # Or 4 rows, 6 cols (24 spots, use 21).
    
    # Let's stick to a dynamic grid based on aspect ratio
    aspect = W / H
    if aspect > 1.5:
        rows, cols = 3, 7
    elif aspect > 1.1:
        rows, cols = 4, 6
    elif aspect > 0.9:
        rows, cols = 5, 5
    elif aspect > 0.6:
        rows, cols = 6, 4
    else:
        rows, cols = 7, 3
        
    total_cells = rows * cols
    if total_cells < N:
        # Fallback to square-ish
        rows = 5
        cols = 5
        total_cells = 25

    v0 = np.zeros(3 * N)
    for i in range(N):
        row = i // cols
        col = i % cols
        cx = (col + 0.5) * W / cols
        cy = (row + 0.5) * H / rows
        # Initial radius: slightly less than half the cell size
        r0 = 0.45 * min(W / cols, H / rows)
        v0[3 * i] = cx
        v0[3 * i + 1] = cy
        v0[3 * i + 2] = r0
        
    return v0

def construct():
    start_time = time.time()
    best_val = 0.0
    best_v = None
    best_WH = (1.0, 1.0)
    
    # Try a range of aspect ratios
    # Perimeter = 4 => W + H = 2
    # W in [0.4, 1.6]
    aspect_ratios = []
    for W in np.linspace(0.5, 1.5, 21):
        H = 2.0 - W
        if H > 0.3:
            aspect_ratios.append((W, H))
            
    # Add specific promising ones
    for W in [0.8, 1.0, 1.2]:
        H = 2.0 - W
        if (W, H) not in aspect_ratios:
            aspect_ratios.append((W, H))
            
    # Sort by closeness to square (often better for packing)
    aspect_ratios.sort(key=lambda wh: abs(wh[0] - wh[1]))
    
    penalty_weight = 100.0
    
    for W, H in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
            
        v0 = _generate_grid_guess(W, H)
        
        # Pre-condition with force relaxation
        v_relaxed = _force_relax(v0, W, H, max_iter=50)
        
        # Optimize using vectorized penalty objective
        # We use L-BFGS-B as it handles large N well and is fast with vectorized gradients (approximated)
        # Or SLSQP with the vectorized objective. L-BFGS-B is often faster for unconstrained-like problems.
        # Since we have bounds, L-BFGS-B is a good fit.
        
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, 0.5 * min(W, H)))
            
        try:
            # We define the objective to capture W, H
            def obj_func(v, W=W, H=H):
                return _vectorized_penalty_objective(v, W, H, penalty_weight)
                
            res = minimize(
                obj_func, v_relaxed, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            v_opt = res.x
            
            # Polish with SLSQP for better constraint satisfaction
            # SLSQP is slower but more accurate on constraints
            def obj_func_slsqp(v, W=W, H=H):
                return _vectorized_penalty_objective(v, W, H, penalty_weight * 10) # Increase penalty for polish
            
            res2 = minimize(
                obj_func_slsqp, v_opt, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            v_final = res2.x
            
            # Validate
            v_valid, val = _validate_and_compute(W, H, v_final)
            if v_valid is not None and val > best_val:
                best_val = val
                best_v = v_valid.copy()
                best_WH = (W, H)
                
        except Exception:
            pass
            
    # Final polish if time remains
    if best_v is not None and time.time() - start_time < TIME_LIMIT - 10:
        W, H = best_WH
        penalty_weight_final = 1000.0
        
        def obj_func_final(v, W=W, H=H):
            return _vectorized_penalty_objective(v, W, H, penalty_weight_final)
            
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, 0.5 * min(W, H)))
            
        try:
            res_final = minimize(
                obj_func_final, best_v, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            v_final = res_final.x
            v_valid, val = _validate_and_compute(W, H, v_final)
            if v_valid is not None and val > best_val:
                best_val = val
                best_v = v_valid.copy()
        except Exception:
            pass
            
    if best_v is not None:
        out = np.zeros((N, 3))
        out[:, 0] = best_v[0::3]
        out[:, 1] = best_v[1::3]
        out[:, 2] = best_v[2::3]
        return out
    else:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(N):
            c = i % side
            r = i // side
            x = (c + 0.5) / side
            y = (r + 0.5) / side
            rad = 0.5 / side - 1e-3
            cells.append([x, y, rad])
        return np.array(cells)

# EVOLVE-BLOCK-END
