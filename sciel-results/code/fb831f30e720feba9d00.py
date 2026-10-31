import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Hexagonal packing in a 1x1 square
    # Scale hex lattice to fit roughly
    # 21 circles in 1x1
    # Hex spacing
    # Rows: y = 0, h, 2h, ...
    # Cols: x = 0, s, 2s, ... offset by s/2 for odd rows
    
    # Let's just use a grid for simplicity and robustness, SLSQP will move them.
    # Grid 5x5, take 21.
    side = 5
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    
    # Create grid points
    points = []
    for y in ys:
        for x in xs:
            points.append([x, y])
            
    # Take first 21
    points = points[:N]
    
    # Initial radii
    # Max radius in 1x1 grid with 5x5 is 0.1
    # Let's start smaller to be safe
    r0 = 0.05
    
    # Variables: W, H, x0, y0, r0, x1, y1, r1, ...
    # W + H = 2
    # Let's fix W=1, H=1 first to simplify, then maybe relax.
    # Or optimize W, H.
    # If we optimize W, H, we need to scale x, y?
    # No, x, y are absolute coordinates.
    # Constraints: 0 <= x_i <= W, 0 <= y_i <= H.
    # And r_i <= x_i, etc.
    
    # Let's try optimizing W, H, x_i, y_i, r_i
    # n_vars = 2 + 3*N
    n_vars = 2 + 3 * N
    
    x0 = np.zeros(n_vars)
    x0[0] = 1.0 # W
    x0[1] = 1.0 # H
    for i in range(N):
        x0[2 + 3*i] = points[i][0]
        x0[3 + 3*i] = points[i][1]
        x0[4 + 3*i] = r0
        
    def objective(vars):
        # Minimize negative sum of radii
        return -np.sum(vars[4::3])
        
    def constraints(vars):
        W, H = vars[0], vars[1]
        # Perimeter constraint: 2(W+H) <= 4 => W+H <= 2
        # We want to maximize, so W+H will likely be 2.
        # Let's enforce W+H=2 to reduce search space?
        # Or W+H <= 2.
        
        cons = []
        # W + H - 2 <= 0
        cons.append({'type': 'ineq', 'fun': lambda v: 2 - v[0] - v[1]})
        
        # Bounds are handled by 'bounds' in minimize, but non-overlap is constraint.
        
        # Non-overlap
        for i in range(N):
            xi = vars[2 + 3*i]
            yi = vars[3 + 3*i]
            ri = vars[4 + 3*i]
            
            # Inside rectangle
            # xi - ri >= 0
            # W - xi - ri >= 0
            # yi - ri >= 0
            # H - yi - ri >= 0
            # These can be in bounds, but SLSQP handles bounds.
            
            for j in range(i+1, N):
                xj = vars[2 + 3*j]
                yj = vars[3 + 3*j]
                rj = vars[4 + 3*j]
                
                # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum_sq = (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[2+3*i]-v[2+3*j])**2 + (v[3+3*i]-v[3+3*j])**2 - (v[4+3*i]+v[4+3*j])**2})
                
        return cons
        
    bounds = []
    # W, H
    bounds.append((0.1, 2.0))
    bounds.append((0.1, 2.0))
    
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
        
    # Run SLSQP
    # Method='SLSQP'
    # maxiter=1000
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            return np.array([[res.x[2+3*i], res.x[3+3*i], res.x[4+3*i]] for i in range(N)])
        else:
            # Fallback to initial grid if failed
            return np.array([[points[i][0], points[i][1], r0] for i in range(N)])
    except Exception as e:
        return np.array([[points[i][0], points[i][1], r0] for i in range(N)])
