import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    best_val = -np.inf
    best_x = None
    
    # Helper to unpack variables
    def unpack(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r

    # Objective function: maximize sum of radii -> minimize negative sum
    def objective(v):
        _, _, r = unpack(v)
        return -np.sum(r)

    def objective_jac(v):
        jac = np.zeros(3 * N)
        jac[2::3] = -1.0
        return jac

    # Constraints
    # We will define a single function that returns all inequality constraints g(v) >= 0
    # and its Jacobian J where J[i, j] = dg_i / dv_j
    
    # Pre-compute pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    num_pairs = len(pairs)
    
    # Total constraints:
    # 4 * N (boundaries: x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0)
    # N * (N-1) / 2 (non-overlap)
    total_cons = 4 * N + num_pairs

    def constraints_fun(v):
        x, y, r = unpack(v)
        g = np.zeros(total_cons)
        
        # Boundary constraints
        # x_i - r_i >= 0
        g[0:N] = x - r
        # 1 - x_i - r_i >= 0
        g[N:2*N] = 1 - x - r
        # y_i - r_i >= 0
        g[2*N:3*N] = y - r
        # 1 - y_i - r_i >= 0
        g[3*N:4*N] = 1 - y - r
        
        # Non-overlap constraints
        idx = 4 * N
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            g[idx] = dist_sq - sum_r*sum_r
            idx += 1
            
        return g

    def constraints_jac(v):
        x, y, r = unpack(v)
        J = np.zeros((total_cons, 3 * N))
        
        # Boundary Jacobians
        # x_i - r_i: d/dx_i = 1, d/dr_i = -1
        for i in range(N):
            J[i, 3*i] = 1.0
            J[i, 3*i+2] = -1.0
            
            # 1 - x_i - r_i: d/dx_i = -1, d/dr_i = -1
            J[N+i, 3*i] = -1.0
            J[N+i, 3*i+2] = -1.0
            
            # y_i - r_i: d/dy_i = 1, d/dr_i = -1
            J[2*N+i, 3*i+1] = 1.0
            J[2*N+i, 3*i+2] = -1.0
            
            # 1 - y_i - r_i: d/dy_i = -1, d/dr_i = -1
            J[3*N+i, 3*i+1] = -1.0
            J[3*N+i, 3*i+2] = -1.0
            
        # Non-overlap Jacobians
        idx = 4 * N
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sum_r = r[i] + r[j]
            
            # d/dx_i = 2*dx, d/dx_j = -2*dx
            J[idx, 3*i] = 2 * dx
            J[idx, 3*j] = -2 * dx
            
            # d/dy_i = 2*dy, d/dy_j = -2*dy
            J[idx, 3*i+1] = 2 * dy
            J[idx, 3*j+1] = -2 * dy
            
            # d/dr_i = -2*sum_r, d/dr_j = -2*sum_r
            J[idx, 3*i+2] = -2 * sum_r
            J[idx, 3*j+2] = -2 * sum_r
            
            idx += 1
            
        return J

    cons = {
        'type': 'ineq',
        'fun': constraints_fun,
        'jac': constraints_jac
    }

    bounds = [(0, 1) for _ in range(3 * N)] # x, y in [0,1], r in [0,1] (r won't exceed 0.5 realistically)
    
    # Multi-start strategy
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10: # Break if close to deadline
            break
            
        # Initialize
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        # Create grid positions
        x_init = np.zeros(N)
        y_init = np.zeros(N)
        r_init = np.zeros(N)
        
        # Place circles in a grid
        for i in range(N):
            row = i // side
            col = i % side
            x_init[i] = (col + 0.5) / side
            y_init[i] = (row + 0.5) / side
            
        # Estimate initial radius
        # Min distance between centers in grid
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x_init[i]-x_init[j])**2 + (x_init[i]-x_init[j])**2) # Wait, y diff
                d = np.sqrt((x_init[i]-x_init[j])**2 + (y_init[i]-y_init[j])**2)
                if d < min_dist:
                    min_dist = d
        # Also check distance to walls
        for i in range(N):
            min_dist = min(min_dist, x_init[i], 1-x_init[i], y_init[i], 1-y_init[i])
            
        r_base = 0.5 * min_dist * 0.9 # 10% safety margin
        
        # Add random jitter
        jitter = 0.05
        x_init += np.random.uniform(-jitter, jitter, N)
        y_init += np.random.uniform(-jitter, jitter, N)
        
        # Clip to valid range to ensure start is somewhat valid
        x_init = np.clip(x_init, 0.05, 0.95)
        y_init = np.clip(y_init, 0.05, 0.95)
        
        v0 = np.zeros(3 * N)
        v0[0::3] = x_init
        v0[1::3] = y_init
        v0[2::3] = r_base
        
        # Run optimization
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                jac=objective_jac, 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )
            
            if res.success:
                v_opt = res.x
                # Check validity
                x, y, r = unpack(v_opt)
                # Verify constraints manually to be safe
                valid = True
                if np.any(r < 0) or np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
                    valid = False
                for i in range(N):
                    if x[i] - r[i] < -1e-6 or x[i] + r[i] - 1 > 1e-6 or y[i] - r[i] < -1e-6 or y[i] + r[i] - 1 > 1e-6:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                
                if valid:
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_x = v_opt.copy()
                        
        except Exception as e:
            pass # Ignore errors in this restart

    if best_x is None:
        # Fallback to seed-like solution if optimization fails
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x, y, r = unpack(best_x)
    result = np.column_stack((x, y, r))
    return result

