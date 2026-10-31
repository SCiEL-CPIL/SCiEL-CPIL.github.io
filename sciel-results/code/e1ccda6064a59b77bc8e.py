import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    # Variables: x1, y1, r1, ..., x21, y21, r21, W, H
    # Total vars: 21*3 + 2 = 65
    
    def objective(z):
        # Minimize negative sum of radii
        r = z[2:63:3]
        return -np.sum(r)

    def constraints(z):
        x = z[0:63:3]
        y = z[1:63:3]
        r = z[2:63:3]
        W = z[63]
        H = z[64]
        
        cons = []
        
        # 1. Perimeter constraint: 2(W+H) <= 4 => W+H <= 2
        cons.append({'type': 'ineq', 'fun': lambda z: 2.0 - (z[63] + z[64])})
        
        # 2. Box constraints
        # x_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
            # x_i + r_i <= W
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[63] - z[3*i] - z[3*i+2]})
            # y_i + r_i <= H
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[64] - z[3*i+1] - z[3*i+2]})
            
        # 3. Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda z, i=i, j=j: 
                            (z[3*i] - z[3*j])**2 + 
                            (z[3*i+1] - z[3*j+1])**2 - 
                            (z[3*i+2] + z[3*j+2])**2
                })
        
        return cons

    # Bounds for variables
    # r_i > 0, x_i, y_i can be anything within box, W, H > 0
    # We set loose bounds to let constraints handle the geometry, 
    # but r_i should be positive.
    bounds = []
    for i in range(N):
        bounds.append((0, 1.0)) # x_i (will be constrained by box)
        bounds.append((0, 1.0)) # y_i
        bounds.append((1e-6, 1.0)) # r_i
    bounds.append((0.1, 2.0)) # W
    bounds.append((0.1, 2.0)) # H

    best_val = -np.inf
    best_z = None

    # Helper to create initial guess
    def make_init(W_init, H_init, offset=0.0):
        # Simple grid initialization
        # Try to fit N circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        # Create a grid of side x side
        xs = []
        ys = []
        rs = []
        
        # Calculate spacing
        # We want to place them roughly in the center
        # Spacing in x: W_init / side
        # Spacing in y: H_init / side
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) * (W_init / side)
                cy = (j + 0.5) * (H_init / side)
                # Initial radius: half the min spacing, slightly smaller to avoid overlap
                r_init = 0.45 * min(W_init / side, H_init / side)
                
                xs.append(cx + offset)
                ys.append(cy + offset)
                rs.append(r_init)
                count += 1
            if count >= N:
                break
        
        # Pad if necessary (shouldn't happen with side=5 for N=21)
        while len(xs) < N:
            xs.append(W_init/2)
            ys.append(H_init/2)
            rs.append(0.1)
            
        z0 = np.zeros(65)
        for i in range(N):
            z0[3*i] = xs[i]
            z0[3*i+1] = ys[i]
            z0[3*i+2] = rs[i]
        z0[63] = W_init
        z0[64] = H_init
        return z0

    # Define several initial aspect ratios and offsets to escape local minima
    # W+H = 2 is the target, so W_init + H_init should be <= 2
    inits = []
    
    # Square
    inits.append((1.0, 1.0, 0.0))
    # Rectangles
    inits.append((1.2, 0.8, 0.0))
    inits.append((0.8, 1.2, 0.0))
    inits.append((1.5, 0.5, 0.0))
    inits.append((0.5, 1.5, 0.0))
    inits.append((1.4, 0.6, 0.0))
    inits.append((1.1, 0.9, 0.0))
    
    # Add some jittered offsets
    np.random.seed(42)
    for _ in range(10):
        w = 0.8 + np.random.rand() * 0.4 # 0.8 to 1.2
        h = 2.0 - w
        off = np.random.rand() * 0.1 - 0.05
        inits.append((w, h, off))

    for W_init, H_init, offset in inits:
        if time.time() > deadline - 10.0:
            break
            
        z0 = make_init(W_init, H_init, offset)
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success:
                val = -res.fun
                # Verify validity and perimeter
                x = res.x[0:63:3]
                y = res.x[1:63:3]
                r = res.x[2:63:3]
                W = res.x[63]
                H = res.x[64]
                
                # Check perimeter
                if W + H <= 2.0 + 1e-5:
                    # Check overlaps
                    valid = True
                    for i in range(N):
                        if x[i] < r[i] - 1e-5 or y[i] < r[i] - 1e-5:
                            valid = False
                            break
                        if x[i] + r[i] > W + 1e-5 or y[i] + r[i] > H + 1e-5:
                            valid = False
                            break
                        for j in range(i+1, N):
                            dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            rad_sum_sq = (r[i]+r[j])**2
                            if dist_sq < rad_sum_sq - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                            
                    if valid and val > best_val:
                        best_val = val
                        best_z = res.x.copy()
                        
        except Exception as e:
            pass

    # If no valid solution found, return a safe grid
    if best_z is None:
        z0 = make_init(1.0, 1.0, 0.0)
        # Shrink radii slightly to ensure validity
        r = z0[2:63:3]
        z0[2:63:3] = r * 0.9
        best_z = z0

    # Format output: array of (x, y, r)
    x = best_z[0:63:3]
    y = best_z[1:63:3]
    r = best_z[2:63:3]
    
    result = np.column_stack((x, y, r))
    return result
