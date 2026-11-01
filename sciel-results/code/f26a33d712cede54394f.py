import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 150.0
START_TIME = time.time()

def objective(params, N=21):
    # params: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, W, H]
    xs = params[0:N]
    ys = params[N:2*N]
    rs = params[2*N:3*N]
    W = params[3*N]
    H = params[3*N+1]
    
    # Sum of radii to maximize
    sum_r = np.sum(rs)
    
    # Penalties
    penalty = 0.0
    
    # 1. Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    perim_viol = max(0.0, W + H - 2.0)
    penalty += 1000.0 * perim_viol**2
    
    # 2. Boundary constraints: circles inside [0,W] x [0,H]
    # x_i - r_i >= 0
    # x_i + r_i <= W
    # y_i - r_i >= 0
    # y_i + r_i <= H
    
    # Using smooth approximations or just quadratic penalties
    for i in range(N):
        # Left
        v1 = max(0.0, rs[i] - xs[i])
        penalty += v1**2
        # Right
        v2 = max(0.0, xs[i] + rs[i] - W)
        penalty += v2**2
        # Bottom
        v3 = max(0.0, rs[i] - ys[i])
        penalty += v3**2
        # Top
        v4 = max(0.0, ys[i] + rs[i] - H)
        penalty += v4**2
        
    # 3. Disjointness: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            req_dist = rs[i] + rs[j]
            # dist >= req_dist
            # penalty if dist < req_dist
            # Using squared distance for smoothness? 
            # dist = sqrt(dist_sq). 
            # If dist_sq < req_dist^2, penalty.
            # Let's use: max(0, req_dist - sqrt(dist_sq + eps))^2
            dist = np.sqrt(dist_sq + 1e-12)
            viol = max(0.0, req_dist - dist)
            penalty += viol**2
            
    return -sum_r + penalty

def construct():
    global START_TIME
    current_time = time.time()
    elapsed = current_time - START_TIME
    if elapsed > DEADLINE - 5.0:
        # Return a valid fallback immediately if out of time
        # Simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-6
                cells.append([x, y, r])
                count += 1
        # Ensure perimeter <= 4. 
        # Bounding box of these circles is approx [0,1]x[0,1] -> Perimeter 4.
        # To be safe, scale down slightly.
        # W=1, H=1. P=4.
        # Scale factor 0.99
        for c in cells:
            c[0] = c[0] * 0.99 + 0.005 # Shift to center
            c[1] = c[1] * 0.99 + 0.005
            c[2] = c[2] * 0.99
        return np.array(cells)

    best_score = -np.inf
    best_params = None
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    init_xs = []
    init_ys = []
    init_rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            init_xs.append((i + 0.5) / side)
            init_ys.append((j + 0.5) / side)
            init_rs.append(0.5 / side)
            count += 1
            
    # Pad if N is not a perfect square grid count? 
    # N=21, side=5. 25 spots. We take first 21.
    # This leaves gaps.
    
    # We will run multiple restarts
    num_restarts = 500 # We will check time in loop
    
    for k in range(num_restarts):
        if time.time() - START_TIME > DEADLINE - 10.0:
            break
            
        # Randomize initial parameters
        # Perturb grid
        xs = np.array(init_xs) + np.random.normal(0, 0.05, N)
        ys = np.array(init_ys) + np.random.normal(0, 0.05, N)
        rs = np.array(init_rs) * (1.0 + np.random.normal(0, 0.1, N))
        
        # Randomize W, H around 1.0
        W = 1.0 + np.random.normal(0, 0.2)
        H = 1.0 + np.random.normal(0, 0.2)
        
        params = np.concatenate([xs, ys, rs, [W, H]])
        
        # Bounds
        # xs, ys can be anywhere, but constrained by penalties
        # rs > 0
        # W, H > 0
        bounds = [(0, 2)] * (2*N) + [(0, 1)] * N + [(0.1, 2), (0.1, 2)]
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < (best_score if best_score != -np.inf else np.inf):
                # Check constraints strictly for the best candidate
                # Extract best params
                p = res.x
                r_sum = np.sum(p[2*N:3*N])
                
                # Verify validity
                # If penalty is low enough, it's likely valid
                if res.fun < -r_sum + 1e-4: # Penalty < 1e-4
                    if r_sum > best_score:
                        best_score = r_sum
                        best_params = p.copy()
                        
        except Exception:
            continue
            
    if best_params is not None:
        p = best_params
        xs = p[0:N]
        ys = p[N:2*N]
        rs = p[2*N:3*N]
        
        # Construct output array
        # The evaluator calculates bounding box from circles.
        # We should ensure the circles are "tight" to the perimeter 4.
        # If W+H < 2, we can scale up.
        # Let's compute the actual bounding box of the solution
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        
        W_act = x_max - x_min
        H_act = y_max - y_min
        P_act = 2 * (W_act + H_act)
        
        if P_act > 4.0:
             # Scale down to fit
             scale = 4.0 / P_act
             # Scale coordinates and radii relative to center
             cx = (x_min + x_max) / 2.0
             cy = (y_min + y_max) / 2.0
             xs = xs * scale + cx * (1 - scale)
             ys = ys * scale + cy * (1 - scale)
             rs = rs * scale
        else:
             # Scale up to use full perimeter?
             # If P_act < 4, we can increase radii?
             # But the optimizer should have pushed them to the limit.
             # If not, we can scale up.
             if P_act > 1e-6:
                 scale = 4.0 / P_act
                 cx = (x_min + x_max) / 2.0
                 cy = (y_min + y_max) / 2.0
                 xs = xs * scale + cx * (1 - scale)
                 ys = ys * scale + cy * (1 - scale)
                 rs = rs * scale
                 
        result = np.column_stack((xs, ys, rs))
        return result
    else:
        # Fallback
        return construct() # Recursive call might be dangerous if time still bad, but we checked.
        # Better to return the grid directly
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([x, y, r])
                count += 1
        return np.array(cells)

# Global timer start
START_TIME = time.time()
