"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def _penalty_function(params, w, h):
    """
    Penalty function for optimization.
    params: [cx_0, ..., cx_{N-1}, cy_0, ..., cy_{N-1}]
    We assume radii are fixed during position optimization.
    """
    cx = params[:N]
    cy = params[N:]
    # Radii are passed via closure or global? Better to use a wrapper.
    # For now, let's assume radii are in a global or passed differently.
    # Actually, minimize takes a function f(x). We need to pass r.
    # We'll use a lambda or a class.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Try different grid inits
        for n_rows in [4, 5]:
            for n_cols in [4, 5]:
                if time.time() > deadline - 15:
                    break
                
                # Create initial positions
                cx = np.zeros(N)
                cy = np.zeros(N)
                r = np.zeros(N)
                
                # Hexagonal-ish grid
                idx = 0
                # Calculate approximate radius
                # Width available per col ~ w / n_cols
                # Height available per row ~ h / n_rows
                r_x = w / (n_cols + 0.5)
                r_y = h / (n_rows + 0.5)
                r_val = min(r_x, r_y) * 0.9 # Start slightly smaller
                
                for i in range(n_rows):
                    for j in range(n_cols):
                        if idx >= N:
                            break
                        x_offset = 0.0 if i % 2 == 0 else r_val * 0.5
                        cx[idx] = r_val + x_offset + j * 2.0 * r_val
                        cy[idx] = r_val + i * 1.732 * r_val # sqrt(3)
                        r[idx] = r_val
                        idx += 1
                
                # Fill remaining with small circles in center
                while idx < N:
                    cx[idx] = w / 2 + np.random.uniform(-0.1, 0.1)
                    cy[idx] = h / 2 + np.random.uniform(-0.1, 0.1)
                    r[idx] = r_val * 0.5
                    idx += 1
                
                # Ensure initial validity by shrinking if needed
                # Simple check
                def is_valid(cx, cy, r, w, h):
                    if np.any(cx - r < -1e-9) or np.any(cx + r - w > 1e-9):
                        return False
                    if np.any(cy - r < -1e-9) or np.any(cy + r - h > 1e-9):
                        return False
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
                            if dist < r[i] + r[j] + 1e-9:
                                return False
                    return True

                if not is_valid(cx, cy, r, w, h):
                    # Shrink radii until valid
                    scale = 1.0
                    while not is_valid(cx, cy, r * scale, w, h) and scale > 0.1:
                        scale *= 0.95
                    r *= scale
                    if not is_valid(cx, cy, r, w, h):
                        continue

                current_score = np.sum(r)
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack((cx, cy, r))

                # Optimization Loop: Expand radii and adjust positions
                # We will use a simple local search: 
                # 1. Try to increase r by 1%
                # 2. Run L-BFGS-B on positions to minimize overlap penalty
                # 3. If valid, accept. Else, break.
                
                for _ in range(50): # Max expansions
                    if time.time() > deadline - 5:
                        break
                    
                    r_new = r * 1.01
                    x0 = np.concatenate([cx, cy])
                    
                    def objective(x):
                        cx_l = x[:N]
                        cy_l = x[N:]
                        # Bounds penalty
                        penalty = 0.0
                        # Out of bounds
                        for i in range(N):
                            if cx_l[i] < r_new[i]:
                                penalty += (r_new[i] - cx_l[i]) ** 2
                            if cx_l[i] > w - r_new[i]:
                                penalty += (cx_l[i] - (w - r_new[i])) ** 2
                            if cy_l[i] < r_new[i]:
                                penalty += (r_new[i] - cy_l[i]) ** 2
                            if cy_l[i] > h - r_new[i]:
                                penalty += (cy_l[i] - (h - r_new[i])) ** 2
                            
                        # Overlap penalty
                        for i in range(N):
                            for j in range(i + 1, N):
                                dist = math.hypot(cx_l[i] - cx_l[j], cy_l[i] - cy_l[j])
                                min_dist = r_new[i] + r_new[j]
                                if dist < min_dist:
                                    penalty += (min_dist - dist) ** 2
                        return penalty

                    # Use L-BFGS-B
                    # We need bounds for x
                    bounds = []
                    for i in range(N):
                        bounds.append((0, w)) # Loose bounds, penalty handles it
                    for i in range(N):
                        bounds.append((0, h))
                        
                    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10})
                    
                    cx_opt = res.x[:N]
                    cy_opt = res.x[N:]
                    
                    # Check validity with strict tolerance
                    if is_valid(cx_opt, cy_opt, r_new, w, h):
                        cx = cx_opt
                        cy = cy_opt
                        r = r_new
                        current_score = np.sum(r)
                        if current_score > best_score:
                            best_score = current_score
                            best_result = np.column_stack((cx, cy, r))
                    else:
                        # If optimization failed to find valid config, try smaller expansion
                        # Or just break
                        break

    # Fallback
    if best_result is None:
        # Simple grid
        side = 5
        cells = []
        w = 1.0
        h = 1.0
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
