import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _force_relax(cx, cy, r, W, H, iterations=200, dt=0.01):
    """
    Force-directed relaxation to resolve overlaps and wall violations.
    Returns updated cx, cy, r.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for _ in range(iterations):
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Wall constraints: push in if touching, push out if overlapping
        for i in range(n):
            # Left wall
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 10
            # Right wall
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            # Bottom wall
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 10
            # Top wall
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
            
            # Radius growth force (we want to maximize radii, but relaxation 
            # should mostly fix positions. We can slightly grow radii if space allows)
            # For now, keep radii fixed during relaxation to ensure feasibility,
            # then optimize radii afterwards.
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-9:
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    f = overlap * 5.0
                    fx[i] += (dx / dist) * f
                    fy[i] += (dy / dist) * f
                    fx[j] -= (dx / dist) * f
                    fy[j] -= (dy / dist) * f
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to hard boundaries to prevent flying out
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
    return cx, cy, r

def _penalty_objective(x, W, H):
    """
    Objective function for L-BFGS-B.
    x = [x1, y1, r1, x2, y2, r2, ...]
    Returns -sum(r) + penalty
    """
    n = N
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Ensure radii are positive
    r = np.maximum(r, 1e-6)
    
    penalty = 0.0
    
    # Wall penalties
    for i in range(n):
        if cx[i] < r[i]:
            penalty += (r[i] - cx[i]) ** 2
        if cx[i] + r[i] > W:
            penalty += (cx[i] + r[i] - W) ** 2
        if cy[i] < r[i]:
            penalty += (r[i] - cy[i]) ** 2
        if cy[i] + r[i] > H:
            penalty += (cy[i] + r[i] - H) ** 2
            
    # Pairwise penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
                
    return -np.sum(r) + 100.0 * penalty

def _optimize_packing(W, H, seed=0):
    """
    Optimize packing for fixed W, H.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with a grid
    # Try different grid layouts
    best_val = -np.inf
    best_x = None
    
    # Layouts: (cols, rows)
    layouts = [(5, 5), (4, 6), (3, 7), (7, 3), (6, 4)]
    
    for cols, rows in layouts:
        if cols * rows < N:
            continue
            
        # Calculate cell size
        cell_w = W / cols
        cell_h = H / rows
        
        cx_init = np.zeros(N)
        cy_init = np.zeros(N)
        r_init = np.zeros(N)
        
        for k in range(N):
            i = k % cols
            j = k // cols
            # Center of cell + small jitter
            cx_init[k] = (i + 0.5) * cell_w + rng.uniform(-0.1, 0.1) * cell_w
            cy_init[k] = (j + 0.5) * cell_h + rng.uniform(-0.1, 0.1) * cell_h
            # Initial radius: slightly less than half the min cell dimension
            r_init[k] = 0.4 * min(cell_w, cell_h)
            
        # Force relax to get a feasible start
        cx_rel, cy_rel, r_rel = _force_relax(cx_init, cy_init, r_init, W, H, iterations=100)
        
        # Construct initial vector
        x0 = np.zeros(3 * N)
        x0[0::3] = cx_rel
        x0[1::3] = cy_rel
        x0[2::3] = r_rel
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, W)) # x
            bounds.append((0, H)) # y
            bounds.append((1e-6, min(W, H)/2)) # r
            
        try:
            res = minimize(_penalty_objective, x0, args=(W, H), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            
            # Check validity
            cx_opt = res.x[0::3]
            cy_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            # Strict validation
            valid = True
            if np.any(cx_opt - r_opt < -1e-5) or np.any(cy_opt - r_opt < -1e-5):
                valid = False
            if np.any(cx_opt + r_opt > W + 1e-5) or np.any(cy_opt + r_opt > H + 1e-5):
                valid = False
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((cx_opt[i]-cx_opt[j])**2 + (cy_opt[i]-cy_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_x = res.x
                    
        except Exception:
            pass
            
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_val = -np.inf
    best_result = None
    
    # We need W + H = 2.
    # Try a range of widths.
    # Symmetry: W and H are interchangeable, so we can just search W in [1, 2] 
    # and treat the rectangle as [0, W] x [0, H].
    # Actually, due to symmetry of the problem (rotation), W in [1, 2] covers all unique aspect ratios.
    
    widths = np.linspace(1.0, 1.8, 15)
    
    for idx, W in enumerate(widths):
        if time.time() > deadline - 5.0:
            break
            
        H = 2.0 - W
        
        # Try a few seeds for this aspect ratio
        # Limit seeds based on time
        seeds_to_try = [0, 1, 2]
        if idx > 5:
            seeds_to_try = [0]
            
        for seed in seeds_to_try:
            if time.time() > deadline - 2.0:
                break
                
            x_opt, val = _optimize_packing(W, H, seed=seed)
            
            if val > best_val:
                best_val = val
                best_result = x_opt
                
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx = best_result[0::3]
    cy = best_result[1::3]
    r = best_result[2::3]
    
    return np.column_stack([cx, cy, r])
# EVOLVE-BLOCK-END
