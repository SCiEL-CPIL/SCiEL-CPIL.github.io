import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    # We will use a multi-start strategy with SLSQP optimization.
    # Each start uses a slightly perturbed grid.
    
    best_sum_radii = -np.inf
    best_solution = None

    # Number of restarts. 120s is a generous time limit. 
    # SLSQP with ~430 constraints and 78 variables might take ~1-5s per run.
    # Let's try 10-20 restarts.
    n_restarts = 15
    
    # Base grid for initialization
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of size side x side
    # We need N circles, so we take the first N from the grid
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Initial radius can be small to ensure no overlap initially
            r = 0.5 / side - 0.05 
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
            
    # Pad if necessary (should not be needed if side*side >= N)
    while count < N:
        xs.append(0.5)
        ys.append(0.5)
        rs.append(0.1)
        count += 1
        
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    rs = np.array(rs[:N])

    for restart in range(n_restarts):
        # Add random perturbation to break symmetry
        if restart > 0:
            perturb = 0.05
            xs_init = xs + np.random.uniform(-perturb, perturb, N)
            ys_init = ys + np.random.uniform(-perturb, perturb, N)
            # Clip to keep them roughly in bounds
            xs_init = np.clip(xs_init, 0.1, 0.9)
            ys_init = np.clip(ys_init, 0.1, 0.9)
            rs_init = rs + np.random.uniform(-0.01, 0.01, N)
            rs_init = np.clip(rs_init, 0.05, 0.4)
        else:
            xs_init = xs.copy()
            ys_init = ys.copy()
            rs_init = rs.copy()

        # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.concatenate([xs_init, ys_init, rs_init])

        # Objective function: Minimize negative sum of radii
        def objective(v):
            r_vals = v[2::3]
            return -np.sum(r_vals)

        # Constraints
        # We need to define inequality constraints g(v) >= 0
        # 1. Boundaries:
        #    x_i - r_i >= 0
        #    1 - x_i - r_i >= 0
        #    y_i - r_i >= 0
        #    1 - y_i - r_i >= 0
        # 2. Overlaps:
        #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0

        def constraints_func(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            cons = []
            
            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x_i - r_i
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]}) # 1 - x_i - r_i
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y_i - r_i
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i
            
            # Overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    # dist^2 - (ri+rj)^2 >= 0
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                    })
            
            return cons

        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5] (max radius in unit square is 0.5)
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x_i
            bounds.append((0, 1)) # y_i
            bounds.append((0.01, 0.5)) # r_i (avoid 0 radius)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_func, options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                v = res.x
                r_vals = v[2::3]
                sum_r = np.sum(r_vals)
                
                # Check validity (safety check)
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                
                valid = True
                # Check bounds
                if np.any(x < -1e-6) or np.any(x > 1+1e-6) or np.any(y < -1e-6) or np.any(y > 1+1e-6):
                    valid = False
                if np.any(r < 0):
                    valid = False
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        sum_r_sq = (r[i]+r[j])**2
                        if dist_sq < sum_r_sq - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break

                if valid and sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_solution = v

        except Exception as e:
            # If optimization fails, continue to next restart
            pass

    if best_solution is None:
        # Fallback to the initial grid if optimization failed completely
        return np.column_stack((xs, ys, rs))
    else:
        x = best_solution[0::3]
        y = best_solution[1::3]
        r = best_solution[2::3]
        return np.column_stack((x, y, r))
