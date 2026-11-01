import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal packing
    # Approximate hexagonal grid
    side = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3)))) # Rough estimate
    # Let's just use a dense grid and pick the best N
    # Hexagonal lattice points
    points = []
    r_hex = 0.05 # Initial small radius
    dx = r_hex * 2
    dy = r_hex * np.sqrt(3)
    for i in range(20):
        for j in range(20):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append((x, y))
    
    # Sort by distance from center to pack centrally? Or just take first N?
    # For hexagonal, it's better to start with a specific pattern.
    # Let's generate a standard hexagonal pack for N=32.
    # Rows of 6, 5, 6, 5 = 22? No.
    # 32 circles.
    # Let's use a simple jittered grid as a reliable fallback.
    
    def make_jittered_grid(n_rows, n_cols, seed=0):
        rng = np.random.RandomState(seed)
        xs = (np.arange(n_cols) + 0.5 + rng.uniform(-0.2, 0.2, n_cols)) / n_cols
        ys = (np.arange(n_rows) + 0.5 + rng.uniform(-0.2, 0.2, n_rows)) / n_rows
        centers = []
        for y in ys:
            for x in xs:
                centers.append((x, y))
        return centers[:N]

    # Generate a few diverse inits
    inits.append(make_jittered_grid(6, 6, seed=1))
    inits.append(make_jittered_grid(8, 4, seed=2))
    inits.append(make_jittered_grid(4, 8, seed=3))
    
    # Hexagonal init
    # Create a hex grid and select 32 closest to center
    hex_pts = []
    for i in range(10):
        for j in range(10):
            x = i * 0.1 + (j % 2) * 0.05
            y = j * 0.0866
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                hex_pts.append((x, y))
    if len(hex_pts) >= N:
        # Sort by distance to center (0.5, 0.5)
        hex_pts.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        inits.append(hex_pts[:N])

    best_score = 0.0
    best_result = None
    best_valid_result = None # Fallback if optimization fails

    def smooth_penalty_objective(params):
        # params: [x0, y0, r0, x1, y1, r1, ...]
        centers = params[0:2*N].reshape(N, 2)
        radii = params[2*N:3*N]
        
        # Objective: Maximize sum(r) - penalty
        # Penalty for overlap
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    pen += overlap * overlap
        
        # Penalty for boundaries
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            # Distance to walls
            d_left = x - r
            d_right = 1 - x - r
            d_bottom = y - r
            d_top = 1 - y - r
            
            viol = 0.0
            if d_left < 0: viol += d_left * d_left
            if d_right < 0: viol += d_right * d_right
            if d_bottom < 0: viol += d_bottom * d_bottom
            if d_top < 0: viol += d_top * d_top
            pen += viol
            
        return -np.sum(radii) + 10.0 * pen

    def is_valid(centers, radii, tol=1e-6):
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True

    for init in inits:
        if time.time() > deadline:
            break
            
        # Convert init to param vector
        # Start with small radii to avoid initial penalty explosion
        init_radii = np.full(N, 0.01)
        init_params = np.concatenate([np.array(init).flatten(), init_radii])
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(smooth_penalty_objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10})
            if time.time() > deadline:
                continue
            best_params = res.x
        except Exception:
            best_params = init_params
            
        # Polish with Powell
        try:
            res_powell = minimize(smooth_penalty_objective, best_params, method='Powell',
                                  options={'maxiter': 200, 'ftol': 1e-12})
            best_params = res_powell.x
        except Exception:
            pass

        # Extract centers and radii
        centers = best_params[0:2*N].reshape(N, 2)
        radii = best_params[2*N:3*N]
        
        # Clip radii to be non-negative
        radii = np.maximum(radii, 0)
        
        # Check validity
        if is_valid(centers, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((centers, radii))
        else:
            # If invalid, try to fix by reducing radii slightly?
            # Or just skip.
            pass

    if best_result is not None:
        return best_result
    else:
        # Fallback: Return the initial grid if optimization failed completely
        centers = np.array(inits[0])
        radii = np.full(N, 0.01)
        # Ensure valid by shrinking
        for i in range(N):
            x, y = centers[i]
            r = min(x, 1-x, y, 1-y, 0.01)
            radii[i] = r
        # Check overlaps and shrink further if needed
        # Simple greedy shrink
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                if radii[i] + radii[j] > max_r:
                    # Shrink the larger one
                    if radii[i] > radii[j]:
                        radii[i] = max(0, max_r - radii[j])
                    else:
                        radii[j] = max(0, max_r - radii[i])
        return np.column_stack((centers, radii))
