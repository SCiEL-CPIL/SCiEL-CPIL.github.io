import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: 5x5 grid, take first 21
    side = 5
    centers = []
    for i in range(side):
        for j in range(side):
            centers.append([ (i + 0.5) / side, (j + 0.5) / side ])
    centers = np.array(centers[:N])
    radii = np.full(N, 0.09) # Slightly smaller than 0.1 to allow movement
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Objective: maximize sum of radii -> minimize negative sum
    def objective(x):
        centers_opt = x[:2*N].reshape(N, 2)
        radii_opt = x[2*N:]
        return -np.sum(radii_opt)
        
    # Constraints:
    # 1. Boundaries: x >= r, y >= r, x <= 1-r, y <= 1-r
    # 2. Non-overlap: dist >= r_i + r_j
    
    def boundary_constraints(x):
        centers_opt = x[:2*N].reshape(N, 2)
        radii_opt = x[2*N:]
        cons = []
        for i in range(N):
            cons.append(centers_opt[i, 0] - radii_opt[i])
            cons.append(centers_opt[i, 1] - radii_opt[i])
            cons.append(1.0 - centers_opt[i, 0] - radii_opt[i])
            cons.append(1.0 - centers_opt[i, 1] - radii_opt[i])
        return cons
        
    def non_overlap_constraints(x):
        centers_opt = x[:2*N].reshape(N, 2)
        radii_opt = x[2*N:]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii_opt[i] + radii_opt[j]
                cons.append(dist_sq - min_dist*min_dist)
        return cons
        
    constraints = []
    # SLSQP expects 'ineq' constraints: f(x) >= 0
    
    # Instead of returning lists, we can define them as functions returning arrays
    # But SLSQP takes a list of dicts {'type': 'ineq', 'fun': lambda}
    
    # It's more efficient to define a single constraint function returning an array
    # However, SLSQP interface for multiple constraints is a list of dicts.
    # For 300 constraints, this is fine.
    
    # Let's try a different approach: Penalty method with gradient descent?
    # Or use 'trust-constr' which can handle array constraints.
    
    # Let's stick to SLSQP with a list of constraints.
    
    def make_boundary(i):
        return lambda x: x[2*i] - x[2*N+i]
    def make_boundary_y(i):
        return lambda x: x[2*i+1] - x[2*N+i]
    def make_boundary_x_max(i):
        return lambda x: 1.0 - x[2*i] - x[2*N+i]
    def make_boundary_y_max(i):
        return lambda x: 1.0 - x[2*i+1] - x[2*N+i]
        
    def make_non_overlap(i, j):
        def fun(x):
            cx_i = x[2*i]
            cy_i = x[2*i+1]
            cx_j = x[2*j]
            cy_j = x[2*j+1]
            r_i = x[2*N+i]
            r_j = x[2*N+j]
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            return dx*dx + dy*dy - (r_i + r_j)**2
        return fun

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': make_boundary(i)})
        cons.append({'type': 'ineq', 'fun': make_boundary_y(i)})
        cons.append({'type': 'ineq', 'fun': make_boundary_x_max(i)})
        cons.append({'type': 'ineq', 'fun': make_boundary_y_max(i)})
        
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': make_non_overlap(i, j)})
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r
        
    best_x = x0
    best_val = objective(x0)
    
    # Try a few random restarts or just one good run
    # SLSQP can be slow. Let's try to run it.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-6})
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        pass

    # Check time
    if time.time() < deadline:
        # Maybe try a second optimization with a different initial guess?
        # Or just return what we have.
        pass

    # Format result
    centers_opt = best_x[:2*N].reshape(N, 2)
    radii_opt = best_x[2*N:]
    
    result = np.zeros((N, 3))
    result[:, 0] = centers_opt[:, 0]
    result[:, 1] = centers_opt[:, 1]
    result[:, 2] = radii_opt
    
    # Final validation and adjustment
    # Ensure no overlaps and within bounds
    # If there are small violations, shrink radii slightly
    for i in range(N):
        # Check bounds
        result[i, 2] = min(result[i, 2], result[i, 0], 1.0 - result[i, 0])
        result[i, 2] = min(result[i, 2], result[i, 1], 1.0 - result[i, 1])
        result[i, 2] = max(result[i, 2], 0)
        
    # Check overlaps and shrink if necessary (simple iterative)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist:
                    # Shrink both radii
                    # Reduce by half the overlap
                    overlap = min_dist - dist
                    # Simple heuristic: shrink both
                    shrink = overlap / 2.0 + 1e-6
                    result[i, 2] -= shrink
                    result[j, 2] -= shrink
                    result[i, 2] = max(result[i, 2], 0)
                    result[j, 2] = max(result[j, 2], 0)
                    changed = True
                    break
            if changed:
                break
    return result
