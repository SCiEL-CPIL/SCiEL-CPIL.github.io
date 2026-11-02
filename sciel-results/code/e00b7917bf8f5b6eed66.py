import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = 110.0
    
    # Container is a 1x1 square (Perimeter = 2*(1+1) = 4)
    W = 1.0
    H = 1.0
    
    # Initial grid placement
    # 5x5 grid has 25 spots. We need 21.
    # Let's use a 5x5 grid and remove the 4 corners? Or just place 21 in a 5x5 grid.
    # A 5x5 grid with spacing 0.2 has radius 0.1. Sum = 21 * 0.1 = 2.1.
    # We can do better.
    
    def get_initial_state():
        # 5 columns, 5 rows. 
        # Indices: i in 0..4, j in 0..4
        # Total 25. We take the first 21?
        # Better: Center a 5x5 block and remove 4 specific ones to make it symmetric?
        # Or just a 4x5 grid (20 circles) + 1?
        # Let's try a 5x5 grid, taking all except 4 corners.
        # Corners: (0,0), (0,4), (4,0), (4,4)
        # Remaining 21.
        xs = []
        ys = []
        rs = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.09 # Slightly less than 0.5/5 = 0.1 to ensure initial validity
                xs.append(x)
                ys.append(y)
                rs.append(r)
        
        if len(xs) != N:
            # Fallback: just fill a grid
            xs = [(i + 0.5) / 5.0 for i in range(5) for j in range(5)]
            ys = [(j + 0.5) / 5.0 for i in range(5) for j in range(5)]
            rs = [0.09] * 25
            xs = xs[:N]
            ys = ys[:N]
            rs = rs[:N]
            
        return np.array(xs), np.array(ys), np.array(rs)

    x0, y0, r0 = get_initial_state()
    
    # Flatten for optimizer
    # Variables: x1, y1, r1, x2, y2, r2, ...
    x_flat = np.concatenate([x0, y0, r0])
    
    def objective_and_penalty(v):
        current_time = time.time()
        if current_time > deadline - 5:
            # Stop optimization, return a high value to signal stop? 
            # Minimize doesn't support easy stopping based on time in callback easily without raising.
            # We will rely on maxiter or a wrapper.
            pass
            
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        
        # 1. Maximize sum of radii -> Minimize -sum(rs)
        val = -np.sum(rs)
        
        # 2. Boundary Penalty
        # dist to boundary: min(x, W-x, y, H-y)
        dist_to_bound = np.minimum(np.minimum(xs, W - xs), np.minimum(ys, H - ys))
        # Penalty if r > dist_to_bound
        over_bound = np.maximum(0, rs - dist_to_bound)
        val += 10.0 * np.sum(over_bound**2)
        
        # 3. Overlap Penalty
        # For each pair i, j
        # d^2 = (xi-xj)^2 + (yi-yj)^2
        # overlap if d < ri + rj
        # Penalty: max(0, (ri+rj) - sqrt(d^2))^2
        # To avoid sqrt for speed, we can use (ri+rj)^2 - d^2 but it's not symmetric in magnitude.
        # Standard soft constraint: max(0, ri + rj - dist)^2
        # Computing all pairs is O(N^2) = 441 ops. Fast enough.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = rs[i] + rs[j]
                if dist < sum_r:
                    diff = sum_r - dist
                    penalty += diff * diff
        val += 100.0 * penalty
        
        return val

    # Constraints on variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in 1x1 box is 0.5)
    bounds = []
    for i in range(N):
        bounds.append((0.0, W)) # x
        bounds.append((0.0, H)) # y
        bounds.append((0.0, 0.5)) # r

    # Optimization
    best_val = np.inf
    best_v = x_flat.copy()
    current_val = objective_and_penalty(x_flat)
    if current_val < best_val:
        best_val = current_val
        best_v = x_flat.copy()

    # Use L-BFGS-B
    # Set a reasonable number of iterations. 
    # N=21, 63 variables.
    # 1000 iterations might be slow if each takes 1ms. 1000ms = 1s.
    # Let's try 500 iterations.
    
    # Callback to update best and check time
    def callback(v):
        nonlocal best_val, best_v
        current_time = time.time()
        if current_time > deadline - 2:
            raise KeyboardInterrupt # Stop optimization
        val = objective_and_penalty(v)
        if val < best_val:
            best_val = val
            best_v = v.copy()
            
    try:
        result = minimize(
            objective_and_penalty, 
            x_flat, 
            method='L-BFGS-B', 
            bounds=bounds,
            callback=callback,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
        )
    except KeyboardInterrupt:
        # Use best_v
        pass
    except Exception as e:
        # Fallback to initial
        pass

    # Extract best solution
    xs = best_v[0:N]
    ys = best_v[N:2*N]
    rs = best_v[2*N:3*N]
    
    # Final Validation and Repair
    # Check if any overlaps or boundary violations
    valid = True
    for i in range(N):
        if xs[i] < -1e-9 or xs[i] > W + 1e-9 or ys[i] < -1e-9 or ys[i] > H + 1e-9:
            valid = False
            break
        if rs[i] > 0.5 + 1e-9 or rs[i] < 0:
            valid = False
            break
    for i in range(N):
        # Boundary
        if rs[i] > xs[i] - 1e-9 or rs[i] > W - xs[i] - 1e-9 or rs[i] > ys[i] - 1e-9 or rs[i] > H - ys[i] - 1e-9:
            valid = False
            break
        # Overlaps
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - 1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to shrink radii to fit
        # This is a simple heuristic: scale down all radii and check
        # Or just fall back to the initial grid which is guaranteed valid
        xs, ys, rs = get_initial_state()
        
    # Construct output array
    circles = np.zeros((N, 3))
    circles[:, 0] = xs
    circles[:, 1] = ys
    circles[:, 2] = rs
    
    # Final sanity check on perimeter
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2 * (bw + bh)
    
    # If perim > 4, it's invalid. 
    # Since we constrained to 1x1 box, perim should be <= 4.
    # But if circles touch boundaries, bw=1, bh=1, perim=4.
    # If circles are inside, bw<1, bh<1, perim<4.
    # The task says "bounding rectangle has perimeter at most 4".
    # So this is valid.
    
    return circles
