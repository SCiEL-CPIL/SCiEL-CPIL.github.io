import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Objective function
    def objective(params):
        pts = params.reshape(N, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        # viol = max(0, r - x) etc.
        viol_x_min = np.maximum(0, r - x)
        viol_x_max = np.maximum(0, r - (1 - x))
        viol_y_min = np.maximum(0, r - y)
        viol_y_max = np.maximum(0, r - (1 - y))
        
        bound_pen = np.sum(viol_x_min**2) + np.sum(viol_x_max**2) + \
                    np.sum(viol_y_min**2) + np.sum(viol_y_max**2)
        
        # Overlap penalties
        # Vectorized pairwise distance calculation
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # We only need i < j
        # Create upper triangular indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[i_idx] + r[j_idx]
        
        # Overlap violation
        viol = np.maximum(0, r_sum - dist)
        overlap_pen = np.sum(viol**2)
        
        # Total penalty
        # Scale penalty to be significant
        penalty_weight = 100.0
        total_pen = penalty_weight * (bound_pen + overlap_pen)
        
        return -sum_r + total_pen

    def is_valid(pts, tol=1e-8):
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r - tol) or np.any(y < r - tol) or np.any(y > 1 - r - tol):
            return False
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[i_idx] + r[j_idx]
        if np.any(dist < r_sum - tol):
            return False
        return True

    def get_value(pts):
        if not is_valid(pts):
            return -np.inf
        return np.sum(pts[:, 2])

    # Initial grid setup
    def make_grid(jitter=0.0):
        side = 6 # 6x6 = 36, take 32
        pts = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01 # small radius initially
                if jitter > 0:
                    cx += np.random.uniform(-jitter, jitter)
                    cy += np.random.uniform(-jitter, jitter)
                pts.append([cx, cy, r])
        return np.array(pts[:N])

    best_val = -np.inf
    best_pts = None
    
    n_restarts = 30
    
    for k in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Vary initial conditions
        jitter = 0.1 * np.random.rand()
        pts = make_grid(jitter=jitter)
        
        # Randomize radii slightly
        pts[:, 2] *= np.random.uniform(0.5, 1.0, N)
        
        # Ensure initial validity roughly
        for i in range(N):
            r = pts[i, 2]
            pts[i, 0] = np.clip(pts[i, 0], r, 1-r)
            pts[i, 1] = np.clip(pts[i, 1], r, 1-r)
            
        params = pts.flatten()
        
        # Use Powell or Nelder-Mead
        # Powell is often faster and more robust for this
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 200, 'disp': False})
            
            if time.time() > deadline - 5:
                continue
                
            res_pts = res.x.reshape(N, 3)
            
            # Post-processing: enforce constraints strictly
            for i in range(N):
                r = res_pts[i, 2]
                res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1-r)
                res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1-r)
                # If radius is too big for the position, shrink radius
                max_r = min(res_pts[i, 0], 1-res_pts[i, 0], res_pts[i, 1], 1-res_pts[i, 1])
                if res_pts[i, 2] > max_r:
                    res_pts[i, 2] = max_r
            
            val = get_value(res_pts)
            if val > best_val:
                best_val = val
                best_pts = res_pts.copy()
                
        except Exception:
            pass

    # Final polish on best result
    if best_pts is not None and time.time() < deadline - 5:
        try:
            params = best_pts.flatten()
            # Use Nelder-Mead for final polish with tighter tol
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            res_pts = res.x.reshape(N, 3)
            for i in range(N):
                r = res_pts[i, 2]
                res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1-r)
                res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1-r)
                max_r = min(res_pts[i, 0], 1-res_pts[i, 0], res_pts[i, 1], 1-res_pts[i, 1])
                if res_pts[i, 2] > max_r:
                    res_pts[i, 2] = max_r
            
            val = get_value(res_pts)
            if val > best_val:
                best_val = val
                best_pts = res_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        side = 6
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 0.01) for i in range(side) for j in range(side)]
        best_pts = np.array(cells[:N])
        
    return best_pts
