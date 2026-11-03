import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = -np.inf
    best_V = None
    
    # Define objective and constraints
    def objective(V):
        rs = V[1+2*N:1+3*N]
        return -np.sum(rs)
        
    def constraints(V):
        W = V[0]
        H = 2.0 - W
        xs = V[1:1+N]
        ys = V[1+N:1+2*N]
        rs = V[1+2*N:1+3*N]
        
        cons = []
        # Boundary
        cons.extend(xs - rs)
        cons.extend(W - xs - rs)
        cons.extend(ys - rs)
        cons.extend(H - ys - rs)
        
        # Pairs
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rad_sum = rs[i] + rs[j]
                cons.append(dist_sq - rad_sum*rad_sum)
                
        return np.array(cons)

    cons = [{'type': 'ineq', 'fun': constraints}]
    bounds = [(0.1, 1.9)] + [(0.0, 2.0)]*N + [(0.0, 2.0)]*N + [(0.0, 1.0)]*N

    # Initializations
    inits = []
    
    # 1. Square Grid
    W0 = 1.0
    xs = np.array([(i+0.5)/5.0 for i in range(5) for _ in range(5)])[:N]
    ys = np.array([(j+0.5)/5.0 for _ in range(5) for j in range(5)])[:N]
    # Note: The list comprehension above needs to be correct for 5x5
    xs_grid = []
    ys_grid = []
    for i in range(5):
        for j in range(5):
            xs_grid.append((i+0.5)/5.0)
            ys_grid.append((j+0.5)/5.0)
    xs = np.array(xs_grid[:N])
    ys = np.array(ys_grid[:N])
    rs = np.full(N, 0.09)
    V0 = np.concatenate([[W0], xs, ys, rs])
    inits.append(V0)
    
    # 2. Hex Grid
    # Rows: 5, 4, 5, 4, 3
    # Spacing dx = 1, dy = sqrt(3)/2
    # We will generate coords and then scale to fit 1x1 box roughly
    hex_xs = []
    hex_ys = []
    rows = [5, 4, 5, 4, 3]
    y_offset = 0
    for r_idx, count in enumerate(rows):
        x_offset = 0.5 if r_idx % 2 == 1 else 0.0
        for k in range(count):
            hex_xs.append(k + x_offset)
            hex_ys.append(y_offset)
        y_offset += np.sqrt(3)/2
        
    hex_xs = np.array(hex_xs[:N])
    hex_ys = np.array(hex_ys[:N])
    
    # Normalize to fit in [0,1]x[0,1]
    min_x, max_x = hex_xs.min(), hex_xs.max()
    min_y, max_y = hex_ys.min(), hex_ys.max()
    # Add padding
    pad = 0.1
    hex_xs = (hex_xs - min_x) / (max_x - min_x + 2*pad) + pad
    hex_ys = (hex_ys - min_y) / (max_y - min_y + 2*pad) + pad
    
    # Estimate radius
    # Min distance between centers
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((hex_xs[i]-hex_xs[j])**2 + (hex_ys[i]-hex_ys[j])**2)
            if d < min_dist: min_dist = d
    r_hex = min_dist / 2.0 * 0.9 # 90% to be safe
    rs_hex = np.full(N, r_hex)
    
    V1 = np.concatenate([[1.0], hex_xs, hex_ys, rs_hex])
    inits.append(V1)
    
    # 3. Random Jittered Square
    V2 = V0.copy()
    np.random.seed(42)
    V2[1:1+N] += np.random.normal(0, 0.02, N)
    V2[1+N:1+2*N] += np.random.normal(0, 0.02, N)
    V2[1+2*N:1+3*N] *= (1 + np.random.normal(0, 0.1, N))
    # Clip to bounds
    V2[1:1+N] = np.clip(V2[1:1+N], 0, 2)
    V2[1+N:1+2*N] = np.clip(V2[1+N:1+2*N], 0, 2)
    V2[1+2*N:1+3*N] = np.clip(V2[1+2*N:1+3*N], 0, 1)
    inits.append(V2)

    # Run optimization
    for V0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, V0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                val = -res.fun
                # Check validity
                cons_val = constraints(res.x)
                if np.min(cons_val) > -1e-6:
                    if val > best_val:
                        best_val = val
                        best_V = res.x
        except Exception as e:
            pass

    # If no valid result found, return a simple valid one
    if best_V is None:
        # Fallback to seed-like
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append([(i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3])
        best_V = np.array(cells[:N])
        # Convert to V format? No, construct returns array of shape (N,3)
        # The optimizer returns V. We need to convert V back to (N,3) array.
        # If we use fallback, it's already (N,3).
        # If we use best_V, we need to convert.
        pass
        
    # Convert best_V to (N,3) array
    if best_V is not None and len(best_V) == 1 + 3*N:
        W = best_V[0]
        xs = best_V[1:1+N]
        ys = best_V[1+N:1+2*N]
        rs = best_V[1+2*N:1+3*N]
        result = np.column_stack((xs, ys, rs))
    else:
        result = best_V
        
    return result
