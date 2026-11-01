import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_sol = None
    
    # Initializations
    inits = []
    
    # 1. Grid 6x6
    side = 6
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    inits.append(np.array(cells[:N]))
    
    # 2. Grid 8x4
    side_x, side_y = 8, 4
    cells = [((i + 0.5) / side_x, (j + 0.5) / side_y) for i in range(side_x) for j in range(side_y)]
    inits.append(np.array(cells[:N]))
    
    # 3. Random
    rng = np.random.default_rng(42)
    inits.append(rng.uniform(0.1, 0.9, (N, 2)))
    
    # 4. Perturbed Grid
    base = inits[0].copy()
    base += rng.normal(0, 0.01, (N, 2))
    inits.append(np.clip(base, 0.05, 0.95))
    
    for init_pos in inits:
        if time.time() > deadline:
            break
            
        # Initial radii: small
        r0 = np.full(N, 0.05)
        x0 = init_pos[:, 0]
        y0 = init_pos[:, 1]
        
        v0 = np.concatenate([x0, y0, r0])
        
        # Objective
        def objective(v):
            r = v[2*N:]
            return -np.sum(r)
            
        # Constraints
        def constraints(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            cons = []
            # Boundaries
            cons.extend([x - r])
            cons.extend([1 - x - r])
            cons.extend([y - r])
            cons.extend([1 - y - r])
            
            # Non-overlap
            # Vectorized pairwise distances
            # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
            # constraint: d_ij^2 - (r_i + r_j)^2 >= 0
            
            # This is O(N^2). For N=32, 496 pairs.
            # SLSQP handles list of constraints.
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    cons.append(dx*dx + dy*dy - dr*dr)
                    
            return cons
            
        # This list of 624 constraints might be slow for SLSQP if not vectorized properly.
        # SLSQP accepts a list of dicts or a function returning an array.
        # Returning an array is faster.
        
        def constraint_func(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            c = np.empty(4*N + N*(N-1)//2)
            idx = 0
            c[idx:idx+N] = x - r
            idx += N
            c[idx:idx+N] = 1 - x - r
            idx += N
            c[idx:idx+N] = y - r
            idx += N
            c[idx:idx+N] = 1 - y - r
            idx += N
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    c[idx] = dx*dx + dy*dy - dr*dr
                    idx += 1
            return c
            
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        # Run SLSQP
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                           constraints=[{'type': 'ineq', 'fun': constraint_func}],
                           options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success:
                v = res.x
                # Check validity
                x, y, r = v[:N], v[N:2*N], v[2*N:]
                valid = True
                if np.any(x < r) or np.any(x > 1-r) or np.any(y < r) or np.any(y > 1-r):
                    valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if d2 < (r[i]+r[j])**2 - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_sol = np.column_stack([x, y, r])
                        
        except Exception:
            pass

    if best_sol is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol
