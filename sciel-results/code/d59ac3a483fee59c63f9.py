import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 26

def _solve_radii_lp(x, y):
    """
    Solve LP to maximize sum of radii given fixed positions.
    Maximize sum(r_i)
    Subject to:
    r_i + r_j <= d_ij
    r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    r_i >= 0
    """
    # Variables: r_0 ... r_{N-1}
    c = -np.ones(N) # Minimize -sum(r)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # r_i + r_j <= d_ij
    # r_i <= bound_i
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints
    for i in range(N):
        bound = min(x[i], 1-x[i], y[i], 1-y[i])
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(bound)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return np.zeros(N)
    except Exception:
        return np.zeros(N)

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_result = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Initializations
    inits = []
    
    # 1. Grid
    grid_pts = []
    for i in range(5):
        for j in range(5):
            if len(grid_pts) < N:
                grid_pts.append([(i+0.5)/5.0, (j+0.5)/5.0])
    # Add 26th
    grid_pts.append([0.5, 0.5])
    inits.append(np.array(grid_pts))
    
    # 2. Hex
    hex_pts = []
    for i in range(6):
        for j in range(5):
            if len(hex_pts) >= N: break
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (i + offset + 0.5) / 6.0
            y = (j + 0.5) / 5.0
            hex_pts.append([x, y])
    inits.append(np.array(hex_pts))
    
    # 3. Random
    rand_pts = rng.uniform(0.1, 0.9, (N, 2))
    inits.append(rand_pts)
    
    # 4. Perturbed Grid
    pert_grid = inits[0].copy()
    pert_grid += rng.uniform(-0.05, 0.05, pert_grid.shape)
    inits.append(pert_grid)
    
    for idx, init_xy in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Initialize radii
        r0 = np.full(N, 0.08)
        x0 = np.concatenate([init_xy.flatten(), r0])
        
        # SLSQP Optimization
        def objective(params):
            return -np.sum(params[2::3])
            
        def objective_grad(params):
            grad = np.zeros_like(params)
            grad[2::3] = -1.0
            return grad
            
        constraints = []
        
        def boundary_constraints(x):
            res = np.zeros(4*N)
            res[0:N] = x[0::3] - x[2::3]
            res[N:2*N] = 1.0 - x[0::3] - x[2::3]
            res[2*N:3*N] = x[1::3] - x[2::3]
            res[3*N:4*N] = 1.0 - x[1::3] - x[2::3]
            return res
            
        def boundary_jac(x):
            J = np.zeros((4*N, 3*N))
            for i in range(N):
                J[i, 3*i] = 1.0
                J[i, 3*i+2] = -1.0
                J[N+i, 3*i] = -1.0
                J[N+i, 3*i+2] = -1.0
                J[2*N+i, 3*i+1] = 1.0
                J[2*N+i, 3*i+2] = -1.0
                J[3*N+i, 3*i+1] = -1.0
                J[3*N+i, 3*i+2] = -1.0
            return J
            
        constraints.append({
            'type': 'ineq',
            'fun': boundary_constraints,
            'jac': boundary_jac
        })
        
        def distance_constraints(x):
            res = []
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = x[3*i:3*i+3]
                    xj, yj, rj = x[3*j:3*j+3]
                    val = (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                    res.append(val)
            return np.array(res)
            
        def distance_jac(x):
            n_con = N*(N-1)//2
            J = np.zeros((n_con, 3*N))
            k = 0
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = x[3*i:3*i+3]
                    xj, yj, rj = x[3*j:3*j+3]
                    J[k, 3*i] = 2*(xi-xj)
                    J[k, 3*i+1] = 2*(yi-yj)
                    J[k, 3*i+2] = -2*(ri+rj)
                    J[k, 3*j] = -2*(xi-xj)
                    J[k, 3*j+1] = -2*(yi-yj)
                    J[k, 3*j+2] = -2*(ri+rj)
                    k += 1
            return J
            
        constraints.append({
            'type': 'ineq',
            'fun': distance_constraints,
            'jac': distance_jac
        })
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                jac=objective_grad,
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            
            if time.time() > deadline - 5:
                continue
                
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            # LP Polish
            r_opt = _solve_radii_lp(x, y)
            
            # Check validity of the polished solution
            valid = True
            for i in range(N):
                if r_opt[i] < 0 or x[i] < r_opt[i] - 1e-6 or x[i] > 1 - r_opt[i] + 1e-6 or y[i] < r_opt[i] - 1e-6 or y[i] > 1 - r_opt[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r_opt[i] + r_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            score = np.sum(r_opt)
            if valid and score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r_opt])
                
        except Exception:
            continue
            
    if best_result is None:
        return np.array(inits[0])
        
    return best_result
