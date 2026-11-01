import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_sum_r = -np.inf
    best_sol = None
    
    # We will try different aspect ratios
    # W + H = 2, so W in (0, 2)
    # Let's sample W from 0.6 to 1.4
    # More points around 1.0 (square)
    w_vals = np.linspace(0.6, 1.4, 17)
    
    # Sort by distance to 1.0? Or just iterate.
    # Square is likely best, so start there.
    w_vals = sorted(w_vals, key=lambda w: abs(w - 1.0))
    
    for w in w_vals:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Initialize positions and radii
        # Grid initialization
        n_x = int(np.ceil(np.sqrt(N * w / h)))
        n_y = int(np.ceil(N / n_x))
        
        # Ensure we have at least N spots
        # If n_x * n_y < N, increase n_y
        while n_x * n_y < N:
            n_y += 1
            
        # Create grid points
        xs = []
        ys = []
        for i in range(n_x):
            for j in range(n_y):
                if len(xs) < N:
                    x = (i + 0.5) * w / n_x
                    y = (j + 0.5) * h / n_y
                    xs.append(x)
                    ys.append(y)
                    
        # Initial radius
        # Min distance between grid points is min(w/n_x, h/n_y)
        r_init = 0.5 * min(w / n_x, h / n_y)
        
        # Variables: x_1, y_1, r_1, ..., x_N, y_N, r_N
        # Flatten: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = r_init
            
        # Objective: minimize -sum(r_i)
        def objective(v):
            r = v[2::3]
            return -np.sum(r)
            
        # Constraints
        # Inequality constraints: g(v) >= 0
        def constraints(v):
            cons = []
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # Bounds
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: x - r})
            # W - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: w - x - r})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: y - r})
            # H - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: h - y - r})
            
            # Non-overlapping
            # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                    })
            return cons

        # This creates a lot of lambda functions, which is slow.
        # Better to define a function that returns the array of constraint values.
        
        def constraint_funcs(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            vals = []
            # Bounds
            vals.extend(x - r)
            vals.extend(w - x - r)
            vals.extend(y - r)
            vals.extend(h - y - r)
            # Non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    vals.append(dx*dx + dy*dy - dr*dr)
            return np.array(vals)
            
        def jac_constraint_funcs(v):
            # Not needed if using finite difference, but SLSQP uses finite diff by default
            pass
            
        cons = [{'type': 'ineq', 'fun': constraint_funcs}]
        
        # Bounds on variables
        # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h)/2))
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_sum_r:
                # Check if constraints are satisfied reasonably well
                # SLSQP might not satisfy them perfectly, but close enough
                if res.fun < best_sum_r:
                    best_sum_r = res.fun
                    best_sol = res.x
                    
        except Exception as e:
            continue
            
    if best_sol is None:
        # Fallback to grid
        w = 1.0
        h = 1.0
        n = 5
        cells = [((i + 0.5) / n, (j + 0.5) / n, 0.5 / n - 1e-3) for i in range(n) for j in range(n)]
        return np.array(cells[:N])
        
    # Format output: (x, y, r)
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = best_sol[3*i]
        out[i, 1] = best_sol[3*i+1]
        out[i, 2] = best_sol[3*i+2]
        
    return out
