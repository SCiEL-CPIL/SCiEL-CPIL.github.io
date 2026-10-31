"""Circle packing in perimeter-4 rectangle: Differential Evolution + L-BFGS refinement."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE

def _penalty_objective(x_flat, W, H, mu=1e4):
    """
    x_flat: flat array of length 3*N [cx0, cy0, r0, cx1, cy1, r1, ...]
    Returns: -sum(r) + penalty(overlaps, boundaries)
    """
    x = x_flat.reshape(-1, 3)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Boundary penalty
    d_left = cx
    d_right = W - cx
    d_bottom = cy
    d_top = H - cy
    
    min_d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    viol_bound = np.maximum(r - min_d, 0.0)
    pen_bound = np.sum(viol_bound ** 2)
    
    # Overlap penalty
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist = np.sqrt(cx_diff**2 + cy_diff**2)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(r_sum, 0)
    
    viol_overlap = np.maximum(r_sum - dist, 0.0)
    pen_overlap = np.sum(viol_overlap ** 2)
    
    # Positive radius penalty
    viol_pos = np.maximum(-r, 0.0)
    pen_pos = np.sum(viol_pos ** 2)
    
    total_penalty = mu * (pen_bound + pen_overlap + pen_pos)
    
    return -np.sum(r) + total_penalty

def _evaluate(x, W, H):
    """
    x: shape (N, 3)
    Returns sum of radii if valid, else -inf
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Check boundaries
    if np.any(cx < -1e-8) or np.any(cx > W + 1e-8) or np.any(cy < -1e-8) or np.any(cy > H + 1e-8):
        return -np.inf
    if np.any(r < 0):
        return -np.inf
        
    # Check overlaps
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist = np.sqrt(cx_diff**2 + cy_diff**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(r_sum, 0)
    
    if np.any(r_sum > dist + 1e-8):
        return -np.inf
        
    return float(np.sum(r))

def _local_refine(x0, W, H):
    """
    Use L-BFGS-B to refine a solution.
    """
    def fun(x_flat):
        return _penalty_objective(x_flat, W, H, mu=1e5)
        
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((1e-6, min(W, H)/2))
        
    try:
        res = minimize(fun, x0.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
        if res.success or res.fun < _penalty_objective(x0.flatten(), W, H, mu=1e5):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return x0

def construct():
    best_sum = -np.inf
    best_arr = None
    
    # Try a range of aspect ratios
    # Perimeter 4 -> 2(W+H)=4 -> W+H=2
    # W in [0.5, 1.5] for reasonable aspect ratios (H in [0.5, 1.5])
    # Square is W=1.
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.77, 1.4, 0.71]
    
    for ar in aspect_ratios:
        if not _time_ok():
            break
            
        # ar = W/H. W = ar*H. W+H=2. H(1+ar)=2 -> H=2/(1+ar), W=2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 - H
        
        # DE Bounds
        de_bounds = []
        for i in range(N):
            de_bounds.append((0, W))
            de_bounds.append((0, H))
            de_bounds.append((0, min(W, H)/2))
            
        # Create a grid initialization
        kx, ky = 5, 5
        cx_grid, cy_grid = np.meshgrid(np.linspace(0.1, W-0.1, kx), np.linspace(0.1, H-0.1, ky))
        cx_flat = cx_grid.flatten()
        cy_flat = cy_grid.flatten()
        r_grid = np.minimum(np.minimum(cx_flat, W-cx_flat), np.minimum(cy_flat, H-cy_flat)) * 0.9
        r_flat = r_grid[:N]
        
        init_arr = np.column_stack([cx_flat[:N], cy_flat[:N], r_flat])
        init_flat = init_arr.flatten()
        
        # Ensure init_flat is within bounds
        init_flat = np.clip(init_flat, [b[0] for b in de_bounds], [b[1] for b in de_bounds])
        
        try:
            res = differential_evolution(
                _penalty_objective, 
                bounds=de_bounds, 
                args=(W, H),
                init=init_flat,
                maxiter=30, 
                popsize=10, 
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if res.x is not None:
                x_de = res.x.reshape(-1, 3)
                # Local refinement
                x_ref = _local_refine(x_de, W, H)
                val = _evaluate(x_ref, W, H)
                if val > best_sum:
                    best_sum = val
                    best_arr = x_ref
                    
        except Exception:
            pass
            
    # Fallback if no valid solution found or time is up
    if best_arr is None or best_sum == -np.inf:
         side = 5
         cells = []
         for i in range(side):
             for j in range(side):
                 if len(cells) < N:
                     cx = (i + 0.5) / side
                     cy = (j + 0.5) / side
                     r = 0.5 / side * 0.9
                     cells.append([cx, cy, r])
         return np.array(cells)
         
    return best_arr
