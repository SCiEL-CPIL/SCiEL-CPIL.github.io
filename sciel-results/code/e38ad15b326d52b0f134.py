"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # We want to pack 21 circles in a rectangle of perimeter 4.
    # Let rectangle dimensions be w and h.
    # Constraint: 2*w + 2*h <= 4  =>  w + h <= 2.
    # To maximize sum of radii, we should likely use the full perimeter, so w + h = 2.
    # We can parameterize by aspect ratio or just optimize w and h directly.
    
    # Initial guess:
    # A square of perimeter 4 has side 1.
    # 21 circles in a 1x1 square.
    # Grid: 5x5 grid has 25 cells. We can use 21 of them.
    # Cell size 0.2. Radius ~ 0.1.
    # Sum of radii ~ 21 * 0.1 = 2.1.
    
    # Let's try to optimize positions, radii, and rectangle dimensions.
    # Variables:
    # x: (N,) centers x
    # y: (N,) centers y
    # r: (N,) radii
    # w: (1,) width
    # h: (1,) height
    # Total variables: 2*N + 2 = 44.
    
    # Objective: maximize sum(r) => minimize -sum(r)
    
    # Constraints:
    # 1. w + h <= 2 (Perimeter constraint)
    # 2. w > 0, h > 0
    # 3. For each circle i:
    #    x[i] - r[i] >= 0
    #    x[i] + r[i] <= w
    #    y[i] - r[i] >= 0
    #    y[i] + r[i] <= h
    # 4. For each pair i < j:
    #    (x[i] - x[j])^2 + (y[i] - y[j])^2 >= (r[i] + r[j])^2
    
    # SLSQP can handle these.
    # However, 21 circles means 210 pair constraints.
    # SLSQP might be slow or get stuck.
    # Let's try a multi-start approach with a good initial guess.
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = v[3*N+1]
        return -np.sum(r)
    
    def constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = v[3*N+1]
        
        cons = []
        
        # Perimeter constraint: w + h - 2 <= 0
        cons.append({'type': 'ineq', 'fun': lambda v, w=w, h=h: 2 - (w + h)})
        
        # Width/Height positivity (handled by bounds mostly, but let's be safe)
        # Actually, SLSQP bounds handle w>0, h>0.
        
        # Circle within rectangle
        for i in range(N):
            # x[i] - r[i] >= 0  =>  x[i] - r[i]
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            # x[i] + r[i] <= w  =>  w - x[i] - r[i]
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            # y[i] - r[i] >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            # y[i] + r[i] <= h
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
            
        # Non-overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                # (x[i]-x[j])^2 + (y[i]-y[j])^2 - (r[i]+r[j])^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        return cons

    # Bounds
    # x, y, r must be positive.
    # w, h must be positive.
    # Max radius is limited by min(w,h)/2.
    # Let's set loose bounds.
    bounds = []
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((0, 1)) # r (max radius in perimeter 4 rect is < 1)
    bounds.append((0.1, 2)) # w
    bounds.append((0.1, 2)) # h
    
    # Initial guess: Grid 5x5 in a 1x1 square.
    # Use first 21 positions of a 5x5 grid.
    side = 5
    x0 = np.zeros(3*N + 2)
    
    # Generate grid positions
    grid_x = []
    grid_y = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
            count += 1
        if count >= N:
            break
            
    x0[:N] = np.array(grid_x)
    x0[N:2*N] = np.array(grid_y)
    x0[2*N:3*N] = 0.5 / side * 0.9 # Slightly smaller radius to start feasible
    x0[3*N] = 1.0 # w
    x0[3*N+1] = 1.0 # h
    
    # Check feasibility of initial guess
    # The grid radii are 0.09. Distance between centers is 0.2.
    # 0.2 >= 0.09 + 0.09 = 0.18. Feasible.
    
    best_val = -np.sum(x0[2*N:3*N])
    best_v = x0.copy()
    
    # Multi-start with slight perturbations if time permits
    # But SLSQP with 210 constraints might be slow.
    # Let's try one run first.
    
    options = {'ftol': 1e-6, 'maxiter': 1000}
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
        if res.success or res.fun < best_val:
            # Check if result is valid
            v = res.x
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            w = v[3*N]
            h = v[3*N+1]
            
            # Verify constraints manually to be safe
            valid = True
            if w + h > 2 + 1e-5:
                valid = False
            for i in range(N):
                if x[i] - r[i] < -1e-5 or x[i] + r[i] > w + 1e-5 or y[i] - r[i] < -1e-5 or y[i] + r[i] > h + 1e-5:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        rad_sum = r[i] + r[j]
                        if dist2 < rad_sum**2 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                best_val = res.fun
                best_v = v.copy()
    except Exception as e:
        pass
    
    # If we have time, try a second start with a different aspect ratio
    if time.time() < deadline - 10:
        # Try a rectangular shape, e.g., w=1.2, h=0.8
        x0_2 = x0.copy()
        x0_2[3*N] = 1.2
        x0_2[3*N+1] = 0.8
        # Rescale positions
        x0_2[:N] = x0[:N] * 1.2
        x0_2[N:2*N] = x0[N:2*N] * 0.8
        # Adjust radii slightly to ensure feasibility
        x0_2[2*N:3*N] *= 0.8
        
        try:
            res2 = minimize(objective, x0_2, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
            if res2.success or res2.fun < best_val:
                v = res2.x
                # Verify validity similarly
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:3*N]
                w = v[3*N]
                h = v[3*N+1]
                valid = True
                if w + h > 2 + 1e-5:
                    valid = False
                for i in range(N):
                    if x[i] - r[i] < -1e-5 or x[i] + r[i] > w + 1e-5 or y[i] - r[i] < -1e-5 or y[i] + r[i] > h + 1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            rad_sum = r[i] + r[j]
                            if dist2 < rad_sum**2 - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_val = res2.fun
                    best_v = v.copy()
        except Exception:
            pass

    # Extract best result
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    w = best_v[3*N]
    h = best_v[3*N+1]
    
    # Final safety check: ensure perimeter <= 4
    if 2*w + 2*h > 4:
        scale = 4.0 / (2*w + 2*h)
        x *= scale
        y *= scale
        r *= scale
        w *= scale
        h *= scale
        
    # Construct output array (x, y, r)
    result = np.column_stack((x, y, r))
    
    return result

# EVOLVE-BLOCK-END
