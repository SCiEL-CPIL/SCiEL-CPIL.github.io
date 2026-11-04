import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    best_value = 0
    best_result = None
    
    # We will try different aspect ratios W (with H = 2 - W)
    # Since W+H=2, W ranges from small to large.
    # Symmetry: W and 2-W are equivalent if we swap x,y. So we can just try W in [1.0, 1.5] or similar.
    # Actually, let's just try a range of W.
    w_values = np.linspace(0.8, 1.2, 10) # H will be 2-W, so H in [0.8, 1.2]
    
    for W in w_values:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Initial guess: Hexagonal lattice
        # Approximate number of circles per row/col
        # For 21 circles, maybe 5 rows of 4? 5*4=20, plus 1?
        # Let's just place them in a grid and let the optimizer move them.
        # A 5x5 grid has 25 spots. We take 21.
        # Or better, generate a dense hexagonal pattern and pick 21 closest to center?
        # Simplest robust init: Grid with jitter.
        
        rows = 5
        cols = 5
        # Create grid of points
        xs = np.linspace(0.1*W, 0.9*W, cols)
        ys = np.linspace(0.1*H, 0.9*H, rows)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.flatten()
        gy = gy.flatten()
        
        # We need 21 points. We have 25. 
        # Let's sort by distance to center and take 21? Or just take first 21?
        # Taking first 21 from a 5x5 grid leaves a corner missing, which is asymmetric.
        # Better: Center the 21 points.
        # Let's use a hexagonal packing init which is denser.
        
        # Generate hexagonal points
        # Side length of hexagon approx radius.
        # Let's just use a random init with a fixed seed for reproducibility, 
        # but try multiple seeds/aspect ratios.
        
        # Actually, for N=21, a 5x5 grid with the 4 corners removed is a very good start.
        # Indices to remove: (0,0), (0,4), (4,0), (4,4) -> 4 corners. 25-4=21.
        
        grid_indices = []
        for r in range(rows):
            for c in range(cols):
                # Skip corners
                if (r==0 and c==0) or (r==0 and c==cols-1) or (r==rows-1 and c==0) or (r==rows-1 and c==cols-1):
                    continue
                grid_indices.append((r,c))
        
        x0 = np.array([xs[c] for r,c in grid_indices])
        y0 = np.array([ys[r] for r,c in grid_indices])
        
        # Initial radii: estimate based on packing density
        # Area of rect = W*H. Area of circles ~ pi * r^2 * N.
        # Packing density ~ 0.9 for hex.
        # N * pi * r^2 ~ 0.9 * W * H
        r_est = np.sqrt(0.9 * W * H / (N * np.pi))
        r0 = np.full(N, r_est * 0.9) # Start a bit smaller to ensure feasibility
        
        x0 = x0 + np.random.uniform(-0.05, 0.05, N) * W # Jitter
        y0 = y0 + np.random.uniform(-0.05, 0.05, N) * H # Jitter
        
        # Variables: x1, y1, r1, x2, y2, r2, ...
        # Total variables: 3 * N = 63
        
        def objective(vars):
            # vars: [x1, y1, r1, x2, y2, r2, ...]
            xs = vars[0::3]
            ys = vars[1::3]
            rs = vars[2::3]
            return -np.sum(rs)
            
        def constraints(vars):
            xs = vars[0::3]
            ys = vars[1::3]
            rs = vars[2::3]
            
            cons = []
            
            # Boundary constraints
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[0::3] - v[2::3]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[1::3] - v[2::3]})
            # W - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: W - v[0::3] - v[2::3]})
            # H - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: H - v[1::3] - v[2::3]})
            
            # Non-overlap constraints
            # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
            for i in range(N):
                for j in range(i+1, N):
                    def non_overlap(v, i=i, j=j):
                        xi = v[3*i]
                        yi = v[3*i+1]
                        ri = v[3*i+2]
                        xj = v[3*j]
                        yj = v[3*j+1]
                        rj = v[3*j+2]
                        return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                    cons.append({'type': 'ineq', 'fun': non_overlap})
                    
            return cons
            
        # Bounds:
        # x in [0, W], y in [0, H], r in [0, max_r]
        # max_r is bounded by min(W, H)/2 roughly, but let's say 1.0
        bounds = []
        for i in range(N):
            bounds.append((0, W)) # x
            bounds.append((0, H)) # y
            bounds.append((0, 1.0)) # r
            
        # Solve
        try:
            res = minimize(objective, 
                           np.concatenate([x0, y0, r0]), 
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to seed-like result if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format result: (x, y, r)
    xs = best_result[0::3]
    ys = best_result[1::3]
    rs = best_result[2::3]
    
    result = np.column_stack((xs, ys, rs))
    return result
