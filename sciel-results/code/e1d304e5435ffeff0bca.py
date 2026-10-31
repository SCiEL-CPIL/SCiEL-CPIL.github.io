"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns an array of shape (26, 3) with rows (x, y, radius).
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n = N
    # Initial guess: a grid arrangement
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    
    x0 = np.array(cells).flatten()
    
    # Define variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    def objective(v):
        return -np.sum(v[2::3])
    
    # Constraints:
    # 1. Circles inside unit square:
    #    x_i >= r_i  => x_i - r_i >= 0
    #    y_i >= r_i  => y_i - r_i >= 0
    #    1 - x_i >= r_i => 1 - x_i - r_i >= 0
    #    1 - y_i >= r_i => 1 - y_i - r_i >= 0
    # 2. Disjoint interiors:
    #    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    cons = []
    
    # Boundary constraints
    for i in range(n):
        xi_idx = 3*i
        yi_idx = 3*i + 1
        ri_idx = 3*i + 2
        
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
    
    # Pairwise disjointness constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })
    
    # Bounds: radii > 0, positions in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((1e-6, 0.5))  # r_i (radius can't exceed 0.5 in unit square)
    
    # Run optimization
    best_result = None
    best_obj = -np.inf
    
    # Try a few random restarts if time permits
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            x_start = x0 + 0.05 * np.random.randn(len(x0))
            # Ensure initial feasibility roughly
            for i in range(n):
                x_start[3*i] = np.clip(x_start[3*i], x_start[3*i+2], 1 - x_start[3*i+2])
                x_start[3*i+1] = np.clip(x_start[3*i+1], x_start[3*i+2], 1 - x_start[3*i+2])
        
        try:
            res = minimize(
                objective, 
                x_start, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
                
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to initial guess if optimization failed
        best_result = x0
    
    # Extract final circles
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = best_result[3*i]
        circles[i, 1] = best_result[3*i+1]
        circles[i, 2] = best_result[3*i+2]
    
    # Final feasibility check and fix
    for i in range(n):
        x, y, r = circles[i]
        # Clip to unit square
        if x < r:
            x = r
        if y < r:
            y = r
        if x > 1 - r:
            x = 1 - r
        if y > 1 - r:
            y = 1 - r
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = r
    
    return circles
# EVOLVE-BLOCK-END
