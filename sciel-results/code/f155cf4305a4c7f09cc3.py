import numpy as np
import scipy.optimize as opt
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 140.0
    
    def objective(z):
        r = z[2::3]
        return -np.sum(r)
    
    # Pre-compute constraint indices for speed
    # We will define constraints as a list of dicts, but we can try to optimize the fun calls
    # Actually, for SLSQP, the fun is called repeatedly.
    
    # Let's try to use a single constraint function that returns all non-overlap constraints
    # and pass it as a single dict? 
    # No, SLSQP requires a list of dicts for 'ineq'.
    # However, we can define one dict with 'fun' returning an array?
    # From docs: "If the constraint is a dictionary, it must have keys 'type' and 'fun'."
    # It does not explicitly say 'fun' can return an array for SLSQP.
    # For 'trust-constr', it can.
    
    # Let's stick to SLSQP with list of constraints, but limit restarts.
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda z, ix=ix, ir=ir: z[ix] - z[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda z, ix=ix, ir=ir: 1 - z[ix] - z[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda z, iy=iy, ir=ir: z[iy] - z[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda z, iy=iy, ir=ir: 1 - z[iy] - z[ir]})
        
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            ix_i, iy_i, ir_i = 3*i, 3*i+1, 3*i+2
            ix_j, iy_j, ir_j = 3*j, 3*j+1, 3*j+2
            def non_overlap(z, ix_i=ix_i, iy_i=iy_i, ir_i=ir_i, ix_j=ix_j, iy_j=iy_j, ir_j=ir_j):
                dx = z[ix_i] - z[ix_j]
                dy = z[iy_i] - z[iy_j]
                dr = z[ir_i] + z[ir_j]
                return dx*dx + dy*dy - dr*dr
            constraints.append({'type': 'ineq', 'fun': non_overlap})
            
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * N
    
    best_result = None
    best_value = -np.inf
    
    # Hexagonal initialization
    def get_hex_guess():
        # Define 26 points in a hexagonal pattern
        # Scale to fit in [0.1, 0.9] to allow room for radii
        points = []
        # Row 0: 5 pts
        for k in range(5):
            points.append((0.1 + 0.2*k, 0.1))
        # Row 1: 6 pts (offset 0.1)
        for k in range(6):
            points.append((0.1 + 0.1 + 0.2*k, 0.1 + 0.17)) # 0.17 approx sqrt(3)/2 * 0.2
        # Row 2: 5 pts
        for k in range(5):
            points.append((0.1 + 0.2*k, 0.1 + 0.34))
        # Row 3: 6 pts
        for k in range(6):
            points.append((0.1 + 0.1 + 0.2*k, 0.1 + 0.51))
        # Row 4: 4 pts
        for k in range(4):
            points.append((0.1 + 0.2*k, 0.1 + 0.68))
            
        # Ensure we have exactly 26
        points = points[:26]
        
        z0 = np.zeros(3*N)
        for i in range(N):
            z0[3*i] = points[i][0]
            z0[3*i+1] = points[i][1]
            z0[3*i+2] = 0.05
        return z0

    # We will run a few restarts with jitter
    for k in range(10):
        if time.time() > deadline - 5:
            break
            
        z0 = get_hex_guess()
        # Add small random jitter
        z0 += np.random.uniform(-0.01, 0.01, size=3*N)
        # Clip to bounds
        z0[0::3] = np.clip(z0[0::3], 0.01, 0.99)
        z0[1::3] = np.clip(z0[1::3], 0.01, 0.99)
        z0[2::3] = np.clip(z0[2::3], 0.01, 0.4)
        
        try:
            res = opt.minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-5})
        except Exception:
            continue
            
        if res.success:
            z = res.x
            # Quick validity check
            valid = True
            for i in range(N):
                x, y, r = z[3*i], z[3*i+1], z[3*i+2]
                if x < r - 1e-4 or x > 1 - r + 1e-4 or y < r - 1e-4 or y > 1 - r + 1e-4:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    if not valid: break
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    for j in range(i+1, N):
                        xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                        if (xi-xj)**2 + (yi-yj)**2 < (ri+rj)**2 - 1e-4:
                            valid = False
                            break
            if valid:
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_result = res.x

    if best_result is None:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.02
                    cells.append([x, y, r])
        return np.array(cells)
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
        
    return result
