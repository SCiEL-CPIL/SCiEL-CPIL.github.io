"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial best solution (grid)
    def get_grid_solution(w, h):
        # Simple grid for N=21. 5x4=20, add 1? Or 4x5?
        # Let's use a 5x5 grid and remove 4? Or just 21 in a 5x5 area.
        # Actually, a hexagonal packing is a better initial guess for the optimizer.
        # Let's generate a hexagonal packing in a 1x1 square.
        circles = []
        # Hexagonal packing parameters
        # Radius r, distance between centers in same row = 2r, vertical = sqrt(3)r
        # Let's try to fit as many as possible in 1x1
        # This is a heuristic initialization.
        
        # Let's just use a grid of 5x5=25 points, take first 21.
        # Centered in 1x1 square.
        side = 5
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius estimate: half the spacing
                r = 0.5 / side * 0.9 
                circles.append([x, y, r])
        return np.array(circles)
    
    best_val = -np.inf
    best_result = None
    
    # Helper to evaluate a solution
    def evaluate_solution(w, circles):
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -np.inf, False
        
        valid = True
        # Check boundaries
        for x, y, r in circles:
            if x < r - 1e-6 or w - x < r - 1e-6 or y < r - 1e-6 or h - y < r - 1e-6:
                valid = False
                break
        if not valid:
            return -np.inf, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                rad_sum = r1 + r2
                if dist_sq < rad_sum**2 - 1e-6:
                    valid = False
                    break
        if not valid:
            return -np.inf, False
            
        val = np.sum(circles[:, 2])
        return val, True

    # Initial guess from grid in a square
    init_circles = get_grid_solution(1.0, 1.0)
    val, valid = evaluate_solution(1.0, init_circles)
    if valid:
        best_val = val
        # Store result in format (x, y, r)
        best_result = np.column_stack((init_circles[:, 0], init_circles[:, 1], init_circles[:, 2]))
        
    # Optimization setup
    # Variables: w, x1..x21, y1..y21, r1..r21
    # Total vars: 1 + 2*21 + 21 = 64
    
    def objective(v):
        w = v[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10
        xs = v[1:1+N]
        ys = v[1+N:1+2*N]
        rs = v[1+2*N:1+3*N]
        
        # Penalty for negative radii or out of bounds
        # We want to maximize sum(r), so minimize -sum(r)
        # Add large penalty for constraint violations to guide SLSQP
        penalty = 0.0
        
        # Boundaries
        for i in range(N):
            if xs[i] < rs[i]:
                penalty += 100 * (rs[i] - xs[i])**2
            if w - xs[i] < rs[i]:
                penalty += 100 * (rs[i] - (w - xs[i]))**2
            if ys[i] < rs[i]:
                penalty += 100 * (rs[i] - ys[i])**2
            if h - ys[i] < rs[i]:
                penalty += 100 * (rs[i] - (h - ys[i]))**2
            if rs[i] < 0:
                penalty += 100 * rs[i]**2
        return -np.sum(rs) + penalty

    # Constraints for SLSQP
    # g(x) >= 0
    def constraints(v):
        w = v[0]
        h = 2.0 - w
        xs = v[1:1+N]
        ys = v[1+N:1+2*N]
        rs = v[1+2*N:1+3*N]
        
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+i] - v[1+2*N+i]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+i] - v[1+2*N+i]}) # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+N+i] - v[1+2*N+i]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[1+N+i] - v[1+2*N+i]}) # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*N+i]}) # r_i >= 0
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[1+i]-v[1+j])**2 + (v[1+N+i]-v[1+N+j])**2 - (v[1+2*N+i]+v[1+2*N+j])**2
                })
        return cons

    # Bounds
    # w in [0.5, 1.5]
    # x in [0, 2], y in [0, 2], r in [0, 1]
    bounds = [(0.5, 1.5)]
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((0, 1)) # r

    # Try a few different initial aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        h_init = 2.0 - w_init
        # Scale initial circles to fit w_init x h_init
        # The grid was for 1x1. Scale x by w_init, y by h_init.
        # But radii also scale? If we scale the whole geometry, radii scale too.
        # Let's just re-initialize the grid for this aspect ratio.
        # For simplicity, use the same grid points but scaled.
        # Actually, let's just perturb the previous best or use the 1x1 grid scaled.
        
        v0 = np.zeros(1 + 3*N)
        v0[0] = w_init
        h_curr = 2.0 - w_init
        
        # Initialize positions in grid for w_init x h_init
        # Use 5x5 grid
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x = (i + 0.5) / side * w_init
                y = (j + 0.5) / side * h_curr
                r = min(w_init, h_curr) / side * 0.4 # Smaller initial radius to avoid overlap
                v0[1+idx] = x
                v0[1+N+idx] = y
                v0[1+2*N+idx] = r
                idx += 1
                
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_val * -1 + 1e-6: # Check if better
                v_res = res.x
                w_res = v_res[0]
                h_res = 2.0 - w_res
                xs = v_res[1:1+N]
                ys = v_res[1+N:1+2*N]
                rs = v_res[1+2*N:1+3*N]
                
                # Construct array
                res_arr = np.column_stack((xs, ys, rs))
                val, valid = evaluate_solution(w_res, res_arr)
                
                if valid and val > best_val:
                    best_val = val
                    best_result = res_arr
                    
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to grid
        return np.array(get_grid_solution(1.0, 1.0))
        
    return best_result

# EVOLVE-BLOCK-END
