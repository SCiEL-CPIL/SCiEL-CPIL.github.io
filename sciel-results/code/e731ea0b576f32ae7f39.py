import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Hexagonal grid in 1x1 square
    # We want to place N circles.
    # A simple way is to create a grid of points, pick the best N, and scale.
    
    def get_hex_grid(n):
        # Generate a hexagonal grid of points
        # This is a heuristic for good initial placement
        rows = int(np.ceil(np.sqrt(n * 1.15))) # Hex packing is denser
        pts = []
        for j in range(rows):
            for i in range(rows):
                x = i * 1.0 + (0.5 if j % 2 else 0.0)
                y = j * 0.866 # sqrt(3)/2
                pts.append((x, y))
        # Sort by distance from center to pick the most central N
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        pts.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        pts = pts[:n]
        
        # Normalize to fit in [0,1]x[0,1]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x if max_x > min_x else 1
        height = max_y - min_y if max_y > min_y else 1
        
        # Scale to fit in 0.9 x 0.9 to leave room for radii
        scale_x = 0.9 / width
        scale_y = 0.9 / height
        scale = min(scale_x, scale_y)
        
        centers = []
        for p in pts:
            x = (p[0] - min_x) * scale + 0.05
            y = (p[1] - min_y) * scale + 0.05
            centers.append([x, y])
        return np.array(centers)

    centers = get_hex_grid(N)
    # Initial radii: small
    r0 = 0.05
    radii = np.full(N, r0)
    
    # Flatten variables: x1, y1, r1, x2, y2, r2, ...
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i][0]
        x0[3*i+1] = centers[i][1]
        x0[3*i+2] = radii[i]

    def unpack(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r

    def objective(v):
        x, y, r = unpack(v)
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(r)

    def constraints(v):
        x, y, r = unpack(v)
        cons = []
        # Boundary constraints: r_i <= x_i <= 1 - r_i
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
        # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })
        
        # Radius positive
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
            
        return cons

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (max radius is 0.5 for 1x1 box)

    best_val = -np.inf
    best_v = x0.copy()
    
    # Multi-start
    num_starts = 5
    
    for k in range(num_starts):
        if time.time() > deadline:
            break
            
        if k == 0:
            v_init = x0
        else:
            # Perturb initial guess
            v_init = x0 + np.random.normal(0, 0.05, 3*N)
            # Clip to bounds
            for i in range(N):
                v_init[3*i] = np.clip(v_init[3*i], 0.01, 0.99)
                v_init[3*i+1] = np.clip(v_init[3*i+1], 0.01, 0.99)
                v_init[3*i+2] = np.clip(v_init[3*i+2], 0.01, 0.4)
        
        try:
            res = minimize(objective, v_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_v = res.x
        except Exception:
            pass
            
    # Check validity and perimeter
    x, y, r = unpack(best_v)
    
    # Verify constraints strictly
    valid = True
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
            valid = False
        for j in range(i+1, N):
            d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            r_sum = r[i] + r[j]
            if d2 < r_sum**2 - 1e-9:
                valid = False
    
    if not valid:
        # Fallback to a simple valid configuration: small circles in grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result

# EVOLVE-BLOCK-END
