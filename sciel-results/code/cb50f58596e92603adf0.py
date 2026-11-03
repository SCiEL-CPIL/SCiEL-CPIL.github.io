"""Rewritten program for circle packing in unit square; SLSQP optimization."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses SLSQP optimization with inequality constraints.
    """
    n = N
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(n)))
    x0_grid = []
    for i in range(side):
        for j in range(side):
            if len(x0_grid) >= n:
                break
            x0_grid.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9])
    x0 = np.array(x0_grid, dtype=float).flatten()
    
    # Objective function: minimize negative sum of radii
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)
    
    # Gradient of objective
    def objective_grad(vars):
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        return grad

    # Constraints
    # 1. Boundary constraints:
    # x_i >= r_i  => x_i - r_i >= 0
    # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i => y_i - r_i >= 0
    # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
    # r_i >= 0.01 (small epsilon to avoid zero radius issues)
    
    def boundary_constraints(vars):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        c = np.zeros(4 * n + n)
        c[0:n] = x - r
        c[n:2*n] = 1 - x - r
        c[2*n:3*n] = y - r
        c[3*n:4*n] = 1 - y - r
        c[4*n:5*n] = r - 0.01
        return c

    def boundary_jacobian(vars):
        # Jacobian shape: (5*n, 3*n)
        J = np.zeros((5 * n, 3 * n))
        idx_x = np.arange(n) * 3
        idx_y = idx_x + 1
        idx_r = idx_x + 2
        
        # x - r >= 0
        J[0:n, idx_x] = 1.0
        J[0:n, idx_r] = -1.0
        
        # 1 - x - r >= 0
        J[n:2*n, idx_x] = -1.0
        J[n:2*n, idx_r] = -1.0
        
        # y - r >= 0
        J[2*n:3*n, idx_y] = 1.0
        J[2*n:3*n, idx_r] = -1.0
        
        # 1 - y - r >= 0
        J[3*n:4*n, idx_y] = -1.0
        J[3*n:4*n, idx_r] = -1.0
        
        # r - 0.01 >= 0
        J[4*n:5*n, idx_r] = 1.0
        
        return J

    # 2. Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def non_overlap_constraints(vars):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        cons = np.zeros((n * (n - 1)) // 2)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                cons[k] = dx**2 + dy**2 - dr**2
                k += 1
        return cons

    def non_overlap_jacobian(vars):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        J = np.zeros(((n * (n - 1)) // 2, 3 * n))
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                
                # Gradient w.r.t x_i: 2*dx
                J[k, 3*i] = 2 * dx
                # Gradient w.r.t y_i: 2*dy
                J[k, 3*i + 1] = 2 * dy
                # Gradient w.r.t r_i: -2*dr
                J[k, 3*i + 2] = -2 * dr
                
                # Gradient w.r.t x_j: -2*dx
                J[k, 3*j] = -2 * dx
                # Gradient w.r.t y_j: -2*dy
                J[k, 3*j + 1] = -2 * dy
                # Gradient w.r.t r_j: -2*dr
                J[k, 3*j + 2] = -2 * dr
                
                k += 1
        return J

    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints, 'jac': boundary_jacobian},
        {'type': 'ineq', 'fun': non_overlap_constraints, 'jac': non_overlap_jacobian}
    ]
    
    # Bounds for variables (helps SLSQP)
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Run SLSQP
    # maxiter and ftol to ensure it finishes within time
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        jac=objective_grad, 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )
    
    # Extract result
    x_opt = result.x[0::3]
    y_opt = result.x[1::3]
    r_opt = result.x[2::3]
    
    # Validate and clean up
    # Ensure strict validity
    valid = True
    eps = 1e-9
    if np.any(x_opt < r_opt - eps) or np.any(x_opt > 1 - r_opt + eps):
        valid = False
    if np.any(y_opt < r_opt - eps) or np.any(y_opt > 1 - r_opt + eps):
        valid = False
    if np.any(r_opt < eps):
        valid = False
        
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2
            min_dist = r_opt[i] + r_opt[j]
            if dist_sq < min_dist**2 - eps:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a simple valid configuration if SLSQP fails or returns invalid
        # Use the grid configuration scaled down slightly
        side = int(np.ceil(np.sqrt(n)))
        x_fallback = []
        y_fallback = []
        r_fallback = []
        for i in range(side):
            for j in range(side):
                if len(x_fallback) >= n:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.95
                x_fallback.append(cx)
                y_fallback.append(cy)
                r_fallback.append(r)
        return np.column_stack([x_fallback, y_fallback, r_fallback])
    
    return np.column_stack([x_opt, y_opt, r_opt])
# EVOLVE-BLOCK-END
