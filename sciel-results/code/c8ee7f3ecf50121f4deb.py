import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Objective: Minimize -sum(r)
    # Variables: [w, x1, y1, r1, ..., xN, yN, rN]
    # h = 2 - w
    
    def unpack(params):
        w = params[0]
        h = 2.0 - w
        circles = params[1:].reshape(N, 3)
        return w, h, circles

    def objective(params):
        w, h, circles = unpack(params)
        # Basic bounds check to avoid invalid log or sqrt issues in constraints if any
        if w <= 0 or h <= 0:
            return 1e10
        radii = circles[:, 2]
        if np.any(radii < 0):
            return 1e10
        return -np.sum(radii)

    def constraints(params):
        w, h, circles = unpack(params)
        cons = []
        
        # Bounds for w
        cons.append({'type': 'ineq', 'fun': lambda p: p[0] - 0.1})
        cons.append({'type': 'ineq', 'fun': lambda p: 1.9 - p[0]})
        
        # Circle bounds
        for i in range(N):
            x, y, r = circles[i]
            # r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[1 + i*3 + 2]})
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[1 + i*3] - p[1 + i*3 + 2]})
            # w - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[0] - p[1 + i*3] - p[1 + i*3 + 2]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[1 + i*3 + 1] - p[1 + i*3 + 2]})
            # h - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[0]) - p[1 + i*3 + 1] - p[1 + i*3 + 2]})
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                # dist^2 - (ri+rj)^2 >= 0
                # dist^2 = (xi-xj)^2 + (yi-yj)^2
                # This is non-linear. SLSQP can handle it.
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                             (p[1+i*3] - p[1+j*3])**2 + (p[1+i*3+1] - p[1+j*3+1])**2 - (p[1+i*3+2] + p[1+j*3+2])**2})
                             
        return cons

    # Initialization: Hexagonal grid
    # Rows: 5, 4, 5, 4, 3
    row_sizes = [5, 4, 5, 4, 3]
    # Estimate radius based on packing in 1x1
    # Hex packing density ~ 0.9069
    # Area 1, 21 circles. Avg area per circle ~ 1/21.
    # r ~ sqrt(1/(21*pi)) ~ 0.123
    # But hex packing is more efficient.
    # Let's try r=0.11 for all, and let optimizer expand.
    
    x0 = [1.0]  # w=1, h=1
    cx = 0.5
    cy = 0.5
    r0 = 0.11
    dx = 2.0 * r0 * np.sqrt(3) / 2  # horizontal spacing
    dy = 2.0 * r0 * 1.5              # vertical spacing
    
    idx = 0
    for i, size in enumerate(row_sizes):
        y = 0.5 + (i - 2) * dy  # Center rows around 0.5
        for j in range(size):
            # Offset every other row
            x_offset = dx / 2 if i % 2 != 0 else 0
            x = 0.5 + (j - (size - 1) / 2.0) * dx + x_offset
            # Ensure x is within bounds roughly
            x0.extend([x, y, r0])
            
    # Check time
    if time.time() > deadline:
        # Return simple grid if no time
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
        
        if time.time() > deadline:
             # Fallback if optimization took too long
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

        if res.success or res.fun < objective(x0):
            w, h, circles = unpack(res.x)
            # Clamp radii to be non-negative and inside box
            result_circles = []
            for i in range(N):
                x, y, r = circles[i]
                r = max(0, r)
                # Ensure inside box
                r = min(r, x, w-x, y, h-y)
                if r > 0:
                    result_circles.append([x, y, r])
                else:
                    # If radius became 0, place at center with tiny radius? 
                    # Or skip? Task says 21 circles.
                    # If r is 0, it's a point. Disjoint interiors is trivially satisfied.
                    # But sum of radii should be maximized.
                    # Let's enforce a minimum radius if it drops to 0 due to numerical error?
                    # Better to just return what we have.
                    result_circles.append([x, y, 1e-6])
            
            return np.array(result_circles)
        else:
            # Fallback to initial guess if optimization failed
            return np.array(x0[1:].reshape(N, 3))
            
    except Exception as e:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
