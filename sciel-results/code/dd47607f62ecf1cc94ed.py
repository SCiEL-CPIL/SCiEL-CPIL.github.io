import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None

    # Define the objective function to minimize: -sum(r)
    def objective(X):
        # X layout: [x_1, y_1, r_1, ..., x_N, y_N, r_N, W]
        W = X[-1]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return 1e10
        r = X[2::3][:N]
        # Simple bound check penalty if needed, but SLSQP handles bounds
        return -np.sum(r)

    def constraints(X):
        W = X[-1]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return [0] * (N*N) # dummy
        cons = []
        x = X[0::3][:N]
        y = X[1::3][:N]
        r = X[2::3][:N]
        
        # Pairwise non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r**2)
        return cons

    def constraint_dict(X):
        W = X[-1]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return {'type': 'ineq', 'fun': lambda: 0}
        def fun(X):
            x = X[0::3][:N]
            y = X[1::3][:N]
            r = X[2::3][:N]
            vals = []
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    sum_r = r[i] + r[j]
                    vals.append(dist_sq - sum_r**2)
            return np.array(vals)
        return {'type': 'ineq', 'fun': fun}

    # Initial guesses
    # 1. Square grid
    # 2. Hex grid
    
    def make_square_grid(W, H):
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                # Center in cell
                cx = (i + 0.5) * (W / side)
                cy = (j + 0.5) * (H / side)
                # Max radius is half min dimension of cell, but we start small
                r = 0.1 * min(W/side, H/side)
                cells.append([cx, cy, r])
        # Pad if necessary
        while len(cells) < N:
            cells.append([W/2, H/2, 0.1])
        return cells

    def make_hex_grid(W, H):
        # Approximate hex packing
        # This is tricky to fit exactly N, so we'll just use a perturbed square grid
        # Or a simple spiral
        cells = []
        cx = W/2
        cy = H/2
        r = min(W, H) / (2 * N) * 0.5
        # Just scatter randomly in the center for diversity
        np.random.seed(42)
        for i in range(N):
            x = np.random.uniform(0.2*W, 0.8*W)
            y = np.random.uniform(0.2*H, 0.8*H)
            cells.append([x, y, r])
        return cells

    # Try different aspect ratios
    for W_init in [0.8, 1.0, 1.2, 1.4]:
        H_init = 2.0 - W_init
        if H_init <= 0: continue
        
        # Generate initial layout
        cells = make_square_grid(W_init, H_init)
        
        # Construct initial X
        X0 = np.zeros(N*3 + 1)
        for i, (cx, cy, r) in enumerate(cells):
            X0[3*i] = cx
            X0[3*i+1] = cy
            X0[3*i+2] = r
        X0[-1] = W_init
        
        # Bounds
        # r > 0, x in [r, W-r], y in [r, H-r]
        # This is coupled. SLSQP bounds are per variable.
        # We can set loose bounds and rely on constraints?
        # No, better to use bounds for r and positions if possible, but positions depend on r.
        # Standard SLSQP bounds:
        # x_i: [0, W_max] -> [0, 2]
        # y_i: [0, H_max] -> [0, 2]
        # r_i: [0, 1]
        # W: [0.1, 1.9]
        
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 2.0)) # x
            bounds.append((1e-6, 2.0)) # y
            bounds.append((1e-6, 1.0)) # r
        bounds.append((0.1, 1.9)) # W
        
        # We need to ensure x_i >= r_i and x_i <= W - r_i.
        # These are inequalities.
        def box_constraints(X):
            W = X[-1]
            H = 2.0 - W
            if W <= 0 or H <= 0: return np.zeros(4*N)
            x = X[0::3][:N]
            y = X[1::3][:N]
            r = X[2::3][:N]
            c = []
            for i in range(N):
                c.append(x[i] - r[i])
                c.append(W - r[i] - x[i])
                c.append(y[i] - r[i])
                c.append(H - r[i] - y[i])
            return np.array(c)

        cons = [constraint_dict(X0), {'type': 'ineq', 'fun': box_constraints}]
        
        try:
            res = minimize(objective, X0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
                
        except Exception as e:
            pass
            
        if time.time() > deadline:
            break

    # Fallback if optimization failed
    if best_solution is None:
        # Return the initial square grid for W=1
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                cx = (i + 0.5) * (W / side)
                cy = (j + 0.5) * (H / side)
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        return np.array(cells)

    X = best_solution
    W = X[-1]
    H = 2.0 - W
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = X[3*i]
        out[i, 1] = X[3*i+1]
        out[i, 2] = X[3*i+2]
        
    return out
