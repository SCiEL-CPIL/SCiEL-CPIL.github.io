"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (i.e., 2*(w+h) = 4 => w+h=2).
    We maximize the sum of radii.
    Returns array of shape (n, 3) where each row is (x, y, r).
    The bounding rectangle of the circles must have perimeter at most 4.
    
    Strategy:
    1. Fix the rectangle dimensions to be a square of side 1.0 (perimeter 4).
       This is a symmetric and often optimal shape for circle packing.
       So w = 1.0, h = 1.0.
    2. Use a grid-based initial guess for circle centers and radii.
    3. Use scipy.optimize.minimize with SLSQP to maximize sum of radii
       subject to:
         - Circle-circle non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
         - Wall constraints: ri <= xi <= 1-ri, ri <= yi <= 1-ri
    4. The variable vector is [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    
    However, SLSQP with 63 variables and ~231 pair constraints + 84 wall constraints
    might be slow or unstable. Let's try a simpler approach first:
    
    Alternative: Use a fixed grid layout that is known to be good.
    For 21 circles in a 1x1 square, a 5x5 grid with some circles removed/adjusted.
    But the seed program used a 5x5 grid with radius 0.5/5 - 1e-3 = 0.0999.
    Sum of radii = 21 * 0.0999 ≈ 2.0979.
    
    Can we do better? Let's try to use scipy.optimize with a penalty method
    or SLSQP with careful constraints.
    
    Given the 150s deadline, let's try a moderate optimization.
    """
    deadline = time.time() + 140  # Safety margin
    
    # Start with a 5x5 grid in a 1x1 square
    side = 5
    xs = np.array([(i + 0.5) / side for i in range(side)])
    ys = np.array([(j + 0.5) / side for j in range(side)])
    
    # Create grid points
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Take first 21 points
    x0 = grid_x[:N]
    y0 = grid_y[:N]
    r0 = np.full(N, 0.5 / side - 1e-3)
    
    # Flatten to single vector
    x0_flat = np.concatenate([x0, y0, r0])
    
    # Define objective: minimize negative sum of radii
    def objective(v):
        r = v[2*N:]
        return -np.sum(r)
    
    # Define constraints
    # Inequality constraints: g(v) >= 0
    # 1. Circle-circle: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    # 2. Wall: xi - ri >= 0, 1 - xi - ri >= 0, yi - ri >= 0, 1 - yi - ri >= 0
    
    def constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        cons = []
        
        # Circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r*sum_r)
        
        # Wall constraints
        for i in range(N):
            cons.append(x[i] - r[i])       # left wall
            cons.append(1 - x[i] - r[i])   # right wall
            cons.append(y[i] - r[i])       # bottom wall
            cons.append(1 - y[i] - r[i])   # top wall
        
        return cons
    
    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0.01, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
    for i in range(N):
        bounds.append((0, 1))  # y
    for i in range(N):
        bounds.append((0.01, 0.5))  # r
    
    # Try SLSQP
    try:
        from scipy.optimize import minimize
        
        # Convert constraints to SLSQP format
        # SLSQP uses 'ineq' for g(v) >= 0
        cons_slsqp = []
        for i in range(N):
            for j in range(i+1, N):
                def circ_constraint(v, i=i, j=j):
                    xi = v[i]
                    xj = v[j + N]
                    yi = v[i + N]
                    yj = v[j + 2*N]
                    ri = v[i + 2*N]
                    rj = v[j + 3*N]
                    dx = xi - xj
                    dy = yi - yj
                    return dx*dx + dy*dy - (ri + rj)**2
                cons_slsqp.append({'type': 'ineq', 'fun': circ_constraint})
        
        for i in range(N):
            def left_wall(v, i=i):
                return v[i] - v[i + 2*N]
            def right_wall(v, i=i):
                return 1 - v[i] - v[i + 2*N]
            def bottom_wall(v, i=i):
                return v[i + N] - v[i + 2*N]
            def top_wall(v, i=i):
                return 1 - v[i + N] - v[i + 2*N]
            
            cons_slsqp.append({'type': 'ineq', 'fun': left_wall})
            cons_slsqp.append({'type': 'ineq', 'fun': right_wall})
            cons_slsqp.append({'type': 'ineq', 'fun': bottom_wall})
            cons_slsqp.append({'type': 'ineq', 'fun': top_wall})
        
        result = minimize(objective, x0_flat, method='SLSQP', bounds=bounds,
                          constraints=cons_slsqp, options={'maxiter': 500, 'ftol': 1e-8})
        
        if result.success:
            x_opt = result.x[:N]
            y_opt = result.x[N:2*N]
            r_opt = result.x[2*N:]
            
            # Check if result is valid
            valid = True
            for i in range(N):
                if r_opt[i] < 0 or x_opt[i] < -1e-6 or x_opt[i] > 1 + 1e-6 or y_opt[i] < -1e-6 or y_opt[i] > 1 + 1e-6:
                    valid = False
                    break
                if x_opt[i] - r_opt[i] < -1e-6 or 1 - x_opt[i] - r_opt[i] < -1e-6:
                    valid = False
                    break
                if y_opt[i] - r_opt[i] < -1e-6 or 1 - y_opt[i] - r_opt[i] < -1e-6:
                    valid = False
                    break
            
            if valid:
                # Clip to ensure strict feasibility
                x_opt = np.clip(x_opt, 0, 1)
                y_opt = np.clip(y_opt, 0, 1)
                r_opt = np.clip(r_opt, 0, 0.5)
                
                # Ensure circles don't overlap by slightly reducing radii if needed
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_opt[i] - x_opt[j]
                        dy = y_opt[i] - y_opt[j]
                        dist = math.sqrt(dx*dx + dy*dy)
                        sum_r = r_opt[i] + r_opt[j]
                        if dist < sum_r:
                            # Reduce both radii slightly
                            factor = dist / sum_r * 0.999
                            r_opt[i] *= factor
                            r_opt[j] *= factor
                
                # Final check and return
                circles = np.column_stack([x_opt, y_opt, r_opt])
                
                # Verify perimeter constraint: bounding box must have perimeter <= 4
                x_min = np.min(x_opt - r_opt)
                x_max = np.max(x_opt + r_opt)
                y_min = np.min(y_opt - r_opt)
                y_max = np.max(y_opt + r_opt)
                w = x_max - x_min
                h = y_max - y_min
                if 2*(w + h) <= 4 + 1e-6:
                    return circles
                else:
                    # Fallback to initial grid
                    pass
        
        # Fallback to initial grid
        circles = np.column_stack([x0, y0, r0])
        return circles
        
    except Exception as e:
        # Fallback to initial grid
        circles = np.column_stack([x0, y0, r0])
        return circles

# EVOLVE-BLOCK-END
