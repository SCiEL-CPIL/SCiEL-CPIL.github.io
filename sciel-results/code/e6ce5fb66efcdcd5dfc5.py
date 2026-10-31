import numpy as np
import scipy.optimize as so
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_score = -1
    best_sol = None
    
    # We will try a few restarts
    # Variables: x (N), y (N), r (N)
    # Fixed w=1, h=1 for now.
    w = 1.0
    h = 1.0
    
    def objective(v):
        # Minimize negative sum of radii
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return -np.sum(r)
    
    def constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        cons = []
        # Wall constraints: r <= x, r <= w-x, r <= y, r <= h-y
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: x - r})
        # w - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: w - x - r})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: y - r})
        # h - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: h - y - r})
        
        # Pairwise constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # This is non-linear. SLSQP can handle it.
        # To make it faster, we can define the constraint function to return an array.
        # However, SLSQP expects a list of dicts or a single fun returning array for 'ineq'.
        # Let's use a single function for all pairwise constraints to reduce overhead.
        
        # We need to be careful with the lambda closures.
        # Better to define a function that takes v and returns the array of constraint values.
        
        return cons

    # The above constraint definition is inefficient because it creates new lambdas every call.
    # Let's rewrite the optimization loop to be more efficient.
    
    def get_constraints_func(w, h):
        def fun(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            c_vals = np.concatenate([
                x - r,
                w - x - r,
                y - r,
                h - y - r
            ])
            
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    c_vals = np.append(c_vals, dx*dx + dy*dy - dr*dr)
            return c_vals
        return fun

    constraint_fun = get_constraints_func(w, h)
    
    bounds = [(0, w) for _ in range(N)] + [(0, h) for _ in range(N)] + [(0, min(w,h)/2) for _ in range(N)]
    
    for restart in range(50): # Max 50 restarts, but break on time
        if time.time() > deadline:
            break
            
        # Initialize
        # Grid initialization with jitter
        side = int(np.ceil(np.sqrt(N)))
        init_x = np.random.uniform(0.1, 0.9, N)
        init_y = np.random.uniform(0.1, 0.9, N)
        init_r = np.full(N, 0.05)
        
        # Better init: Grid
        idx = 0
        init_x = np.zeros(N)
        init_y = np.zeros(N)
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                init_x[idx] = (i + 0.5) / side
                init_y[idx] = (j + 0.5) / side
                idx += 1
        # Add jitter
        init_x += np.random.uniform(-0.02, 0.02, N)
        init_y += np.random.uniform(-0.02, 0.02, N)
        # Clip to bounds
        init_x = np.clip(init_x, 0.05, w-0.05)
        init_y = np.clip(init_y, 0.05, h-0.05)
        
        v0 = np.concatenate([init_x, init_y, init_r])
        
        try:
            res = so.minimize(objective, v0, method='SLSQP', bounds=bounds, 
                               constraints=[{'type': 'ineq', 'fun': constraint_fun}], 
                               options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                v_opt = res.x
                score = -res.fun
                # Verify validity
                x = v_opt[:N]
                y = v_opt[N:2*N]
                r = v_opt[2*N:3*N]
                
                valid = True
                # Check walls
                if np.any(x < r - 1e-6) or np.any(w-x < r - 1e-6) or np.any(y < r - 1e-6) or np.any(h-y < r - 1e-6):
                    valid = False
                # Check pairs
                for i in range(N):
                    for j in range(i+1, N):
                        d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if d2 < (r[i]+r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and score > best_score:
                    best_score = score
                    best_sol = v_opt
                    
        except Exception as e:
            continue
            
    if best_sol is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format output
    x = best_sol[:N]
    y = best_sol[N:2*N]
    r = best_sol[2*N:3*N]
    
    return np.column_stack((x, y, r))

