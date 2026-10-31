"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial guess: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([(i + 0.5) / side, (j + 0.5) / side])
    
    # We'll optimize positions and radii.
    # Variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    # Constraints:
    # 1. r_i >= 0
    # 2. x_i >= r_i
    # 3. y_i >= r_i
    # 4. x_i + r_i <= 1
    # 5. y_i + r_i <= 1
    # 6. ||p_i - p_j||_2 >= r_i + r_j  =>  (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    
    # For scipy.optimize.minimize with SLSQP, we can handle inequality constraints.
    # However, non-smoothness of the distance constraint gradient at 0 can be an issue.
    # We can use a smooth approximation or just rely on SLSQP's robustness.
    # Actually, the constraint g(i,j) = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0 is smooth.
    
    def objective(v):
        # v is flattened [x1, y1, r1, x2, y2, r2, ...]
        radii = v[2::3]
        return -np.sum(radii)
    
    def objective_grad(v):
        grad = np.zeros_like(v)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(N):
        xi = 3*i
        yi = 3*i + 1
        ri = 3*i + 2
        
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([1.0, 0.0, -1.0])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([0.0, 1.0, -1.0])
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.array([-1.0, 0.0, -1.0])
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.array([0.0, -1.0, -1.0])
        })
        # r_i >= 0 (handled by bounds, but also as constraint for safety if needed)
        # Actually, bounds are better for r_i >= 0.
        
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2,
                'jac': lambda v, i=i, j=j: _dist_jac(v, i, j)
            })
    
    def _dist_jac(v, i, j):
        jac = np.zeros_like(v)
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        r_sum = v[3*i+2] + v[3*j+2]
        
        # grad of (dx^2 + dy^2 - r_sum^2)
        # d/dx_i = 2*dx
        # d/dy_i = 2*dy
        # d/dx_j = -2*dx
        # d/dy_j = -2*dy
        # d/dr_i = -2*r_sum
        # d/dr_j = -2*r_sum
        
        jac[3*i] = 2*dx
        jac[3*i+1] = 2*dy
        jac[3*j] = -2*dx
        jac[3*j+1] = -2*dy
        jac[3*i+2] = -2*r_sum
        jac[3*j+2] = -2*r_sum
        return jac

    # Initial vector
    v0 = np.zeros(N*3)
    for i in range(N):
        v0[3*i] = positions[i][0]
        v0[3*i+1] = positions[i][1]
        v0[3*i+2] = 0.1  # Small initial radius
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5)) # r (must be >0, max 0.5)
    
    best_result = None
    best_value = -np.inf
    
    # Try multiple restarts with different initializations if time permits
    # But first, try the grid initialization
    
    try:
        res = minimize(objective, v0, method='SLSQP', jac=objective_grad,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success or res.fun < best_value:
            best_result = res.x
            best_value = -res.fun
            
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for more, return current best
        if best_result is not None:
            return _format_result(best_result)
        else:
            return _format_result(v0)
    
    # Try a second initialization: random
    np.random.seed(42)
    v1 = np.zeros(N*3)
    for i in range(N):
        v1[3*i] = np.random.uniform(0.1, 0.9)
        v1[3*i+1] = np.random.uniform(0.1, 0.9)
        v1[3*i+2] = np.random.uniform(0.05, 0.15)
        
    try:
        res2 = minimize(objective, v1, method='SLSQP', jac=objective_grad,
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res2.success or res2.fun < best_value:
            best_result = res2.x
            best_value = -res2.fun
            
    except Exception as e:
        pass
    
    if time.time() > deadline - 5:
        if best_result is not None:
            return _format_result(best_result)
        else:
            return _format_result(v0)
    
    # Try a third initialization: clustered
    v2 = np.zeros(N*3)
    # Place in a tighter grid to encourage larger radii
    side2 = 5
    for i in range(N):
        row = i // side2
        col = i % side2
        v2[3*i] = (col + 0.5) / side2
        v2[3*i+1] = (row + 0.5) / side2
        v2[3*i+2] = 0.15
        
    try:
        res3 = minimize(objective, v2, method='SLSQP', jac=objective_grad,
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res3.success or res3.fun < best_value:
            best_result = res3.x
            best_value = -res3.fun
            
    except Exception as e:
        pass
        
    if best_result is not None:
        return _format_result(best_result)
    else:
        return _format_result(v0)

def _format_result(v):
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = v[3*i]
        result[i, 1] = v[3*i+1]
        result[i, 2] = v[3*i+2]
    return result
# EVOLVE-BLOCK-END
