import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    # Variables: [x1, y1, r1, ..., xN, yN, rN, W, H]
    # Total 3*N + 2 variables
    
    def objective_and_grad(v):
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            raise TimeoutError()
            
        # Unpack
        circles = v[:3*N].reshape(N, 3)
        W = v[3*N]
        H = v[3*N+1]
        
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Ensure positive radii and dimensions
        r = np.maximum(r, 1e-6)
        W = np.maximum(W, 1e-6)
        H = np.maximum(H, 1e-6)
        
        # Sum of radii to maximize
        sum_r = np.sum(r)
        
        # Penalties
        penalty = 0.0
        
        # Boundary constraints:
        # x >= r, x <= W-r
        # y >= r, y <= H-r
        # W+H <= 2
        
        # Left
        viol_l = np.maximum(0, r - x)
        # Right
        viol_r = np.maximum(0, x - (W - r))
        # Bottom
        viol_b = np.maximum(0, r - y)
        # Top
        viol_t = np.maximum(0, y - (H - r))
        
        boundary_viol = np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2)
        
        # Perimeter constraint: 2(W+H) <= 4 => W+H <= 2
        perim_viol = np.maximum(0, W + H - 2.0)**2
        
        # Overlap constraints
        # d_ij >= r_i + r_j
        overlap_viol = 0.0
        # Vectorized pairwise distance calculation is efficient for N=21
        # d^2 = (xi-xj)^2 + (yi-yj)^2
        # We need d < ri+rj
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # Mask upper triangle to avoid double counting and self-overlap
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_upper = d[mask]
        r_sum_upper = r_sum[mask]
        
        viol_dist = np.maximum(0, r_sum_upper - d_upper)
        overlap_viol = np.sum(viol_dist**2)
        
        total_viol = boundary_viol + perim_viol + overlap_viol
        
        # Objective: Minimize -sum_r + lambda * violation
        # Lambda should be large enough to enforce constraints
        lam = 100.0
        obj = -sum_r + lam * total_viol
        
        # Gradient calculation (numerical or analytical)
        # For speed and simplicity in this context, we can let L-BFGS-B do numerical gradient
        # But analytical is better. However, implementing gradient for penalty is complex.
        # Let's return objective only and let scipy use numerical gradient.
        # Given N=21 (65 vars), numerical gradient is acceptable if not too many iterations.
        # To be safe on time, we might stick to objective only.
        
        return obj

    def initial_guess_grid(rows, cols):
        # Create a grid of circles
        W_init = 1.0
        H_init = 1.0
        # Adjust aspect ratio if needed? Let's start with 1x1
        # We need N circles.
        # If rows*cols >= N, we place N circles in the first N spots of the grid
        # And remove the rest? No, we need exactly N circles.
        # So we create a grid of size rows x cols, and if rows*cols > N, we just use the first N?
        # That leaves empty space. Better to distribute them?
        # Simplest: Create a grid that fits N.
        # e.g. 5x5 = 25. Use 21.
        # Let's just use a dense grid and let the optimizer remove/merge? No, count is fixed.
        # We initialize with a grid pattern.
        
        xs = []
        ys = []
        rs = []
        
        # Calculate cell size
        # Let's assume a square 1x1 for init
        # We want to place N circles.
        # Let's try to find a grid size that is close to sqrt(N)
        # Or just use a specific layout.
        
        # Layout 1: 5x5 grid, take first 21
        # Center of cell (i, j) in a 5x5 grid in 1x1 box
        # x = (i + 0.5) / 5
        # y = (j + 0.5) / 5
        # r = 0.5 / 5 = 0.1
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N:
                    break
                x = (i + 0.5) / rows
                y = (j + 0.5) / cols
                # Radius should be small enough to not overlap initially?
                # In a grid, max radius is 0.5/rows or 0.5/cols.
                # Let's use a slightly smaller radius to ensure no initial overlap penalty
                r = 0.45 / max(rows, cols)
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
            if count >= N:
                break
        
        # If we didn't fill N (should not happen if rows*cols >= N)
        while len(xs) < N:
            xs.append(0.5)
            ys.append(0.5)
            rs.append(0.05)
            
        v = np.concatenate([np.array(xs), np.array(ys), np.array(rs), [1.0, 1.0]])
        return v

    best_obj = np.inf
    best_v = None
    
    # Multi-start strategies
    starts = []
    
    # 1. 5x5 Grid (21 of 25)
    starts.append(initial_guess_grid(5, 5))
    
    # 2. 3x7 Grid (21 of 21)
    starts.append(initial_guess_grid(3, 7))
    
    # 3. 7x3 Grid
    starts.append(initial_guess_grid(7, 3))
    
    # 4. Random
    for _ in range(5):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.random.rand(N) * 0.1 + 0.05 # Small radii
        W = 1.0
        H = 1.0
        v = np.concatenate([x, y, r, [W, H]])
        starts.append(v)
        
    # 5. Square packing heuristic (Hexagonal)
    # Approximate hexagonal packing in 1x1
    # This is complex to generate exactly 21, so we rely on random/grid
    
    for v0 in starts:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        try:
            res = minimize(objective_and_grad, v0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_obj:
                # Check validity of the result
                v_res = res.x
                circles = v_res[:3*N].reshape(N, 3)
                W = v_res[3*N]
                H = v_res[3*N+1]
                
                x = circles[:, 0]
                y = circles[:, 1]
                r = circles[:, 2]
                
                # Check if constraints are satisfied (with small tolerance)
                tol = 1e-5
                valid = True
                
                # Boundary
                if np.any(x < r - tol) or np.any(y < r - tol):
                    valid = False
                if np.any(x > W - r + tol) or np.any(y > H - r + tol):
                    valid = False
                
                # Perimeter
                if W + H > 2.0 + tol:
                    valid = False
                
                # Overlap
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d = np.sqrt(dx**2 + dy**2)
                r_sum = r[:, None] + r[None, :]
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                if np.any(d[mask] < r_sum[mask] - tol):
                    valid = False
                
                if valid:
                    best_obj = res.fun
                    best_v = v_res
                else:
                    # If invalid, try to repair by shrinking radii or moving inward?
                    # Or just skip if penalty wasn't high enough.
                    # Given high lambda, it should be valid.
                    # If not, we can scale down radii to fix overlaps.
                    pass
                    
        except Exception as e:
            continue

    if best_v is None:
        # Fallback to a very safe, small grid
        # 5x5 grid with tiny radii
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.05
                cells.append([x, y, r])
                count += 1
            if count >= N:
                break
        return np.array(cells)
        
    # Extract best result
    circles = best_v[:3*N].reshape(N, 3)
    # Ensure strict validity for the return value
    # Clip radii to be positive
    circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
    
    # Final repair pass: if any overlap, shrink radii slightly
    # This ensures the returned array is strictly valid
    for iter in range(10):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        W = best_v[3*N]
        H = best_v[3*N+1]
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        max_viol = 0
        for i in range(N):
            for j in range(i+1, N):
                if d[i,j] < r[i] + r[j]:
                    viol = (r[i] + r[j] - d[i,j]) / 2
                    r[i] -= viol
                    r[j] -= viol
                    max_viol = max(max_viol, viol)
        
        # Check boundaries
        for i in range(N):
            if x[i] < r[i]: r[i] = x[i]
            if y[i] < r[i]: r[i] = y[i]
            if W - x[i] < r[i]: r[i] = W - x[i]
            if H - y[i] < r[i]: r[i] = H - y[i]
            r[i] = max(r[i], 1e-6)
            
        if max_viol < 1e-8:
            break
            
    circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
    return circles

# Ensure construct is callable and returns the right shape
# The function is defined above.
