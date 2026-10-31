"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Initial configuration: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    
    # Convert to numpy array
    circles = np.array(cells[:N])
    
    # Objective function: minimize negative sum of radii (maximize sum of radii)
    # with penalties for violations
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Sum of radii (we minimize negative)
        neg_sum_r = -np.sum(r)
        
        # Boundary violations: circle must be inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        bound_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                     np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        
        # Overlap violations: distance between centers >= sum of radii
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_viol += (min_dist - dist)
        
        # Penalty weights
        weight_bound = 10.0
        weight_overlap = 10.0
        
        return neg_sum_r + weight_bound * np.sum(bound_viol) + weight_overlap * overlap_viol
    
    # Constraints for SLSQP
    def boundary_constraint_x(params):
        x = params[:N]
        r = params[2*N:3*N]
        return x - r  # >= 0
    
    def boundary_constraint_x1(params):
        x = params[:N]
        r = params[2*N:3*N]
        return (1 - x) - r  # >= 0
    
    def boundary_constraint_y(params):
        y = params[N:2*N]
        r = params[2*N:3*N]
        return y - r  # >= 0
    
    def boundary_constraint_y1(params):
        y = params[N:2*N]
        r = params[2*N:3*N]
        return (1 - y) - r  # >= 0
    
    # Overlap constraints
    def make_overlap_constraint(i, j):
        def constraint(params):
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - r[i] - r[j]  # >= 0
        return constraint
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraint_x},
        {'type': 'ineq', 'fun': boundary_constraint_x1},
        {'type': 'ineq', 'fun': boundary_constraint_y},
        {'type': 'ineq', 'fun': boundary_constraint_y1},
    ]
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
    for i in range(N):
        bounds.append((0, 1))  # y
    for i in range(N):
        bounds.append((0, 0.5))  # r
    
    # Initial parameters
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Run optimization with time check
    best_result = None
    best_value = -np.inf
    
    # First pass: L-BFGS-B with penalty method
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12})
        
        if result.success or result.fun < best_value:
            best_result = result
            best_value = result.fun
        
        if time.time() - start_time > deadline - safety_margin:
            # Return current best
            if best_result is not None:
                params = best_result.x
                x = params[:N]
                y = params[N:2*N]
                r = params[2*N:3*N]
                # Clamp to valid range
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                r = np.clip(r, 0, 0.5)
                return np.column_stack([x, y, r])
            return circles
    except Exception:
        pass
    
    # Second pass: SLSQP with constraints for higher precision
    if time.time() - start_time < deadline - safety_margin - 10:
        try:
            x0_slsqp = best_result.x if best_result is not None else x0
            result2 = minimize(lambda p: -np.sum(p[2*N:3*N]), x0_slsqp,
                             method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 500, 'ftol': 1e-12})
            
            if result2.success or result2.fun < best_value:
                best_result = result2
                best_value = result2.fun
        except Exception:
            pass
    
    # Extract best result
    if best_result is not None:
        params = best_result.x
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        # Clamp to valid range
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0, 0.5)
        return np.column_stack([x, y, r])
    else:
        return circles
# EVOLVE-BLOCK-END
