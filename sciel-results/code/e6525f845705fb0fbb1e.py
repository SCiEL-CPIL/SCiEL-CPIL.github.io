"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
import itertools

def _objective_and_constraints(x, n):
    """
    x: flattened array of length 3n: [x0,y0,r0, x1,y1,r1, ...]
    Returns (objective, constraints_list)
    Objective: -sum(r) (we maximize sum r, so minimize -sum r)
    Constraints:
      - r_i >= 0
      - x_i >= r_i
      - x_i <= 1 - r_i
      - y_i >= r_i
      - y_i <= 1 - r_i
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  for all i < j
    """
    n = x.shape[0] // 3
    pts = x.reshape(n, 3)
    radii = pts[:, 2]
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    obj = -np.sum(radii)
    
    cons = []
    
    # Boundary and positivity constraints
    # r_i >= 0
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # We'll build a linear constraint matrix for the linear ones
    # A_ub x <= b_ub
    # But SLSQP uses ineq: c(x) >= 0
    # So: r_i >= 0 -> c = r_i
    #     x_i - r_i >= 0 -> c = x_i - r_i
    #     1 - x_i - r_i >= 0 -> c = 1 - x_i - r_i
    #     y_i - r_i >= 0 -> c = y_i - r_i
    #     1 - y_i - r_i >= 0 -> c = 1 - y_i - r_i
    
    # There are 5n linear inequality constraints.
    # We can group them.
    
    # Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    for i in range(n):
        xi, yi, ri = pts[i]
        cons.append(ri)  # r_i >= 0
        cons.append(xi - ri)  # x_i >= r_i
        cons.append(1 - xi - ri)  # x_i <= 1 - r_i
        cons.append(yi - ri)  # y_i >= r_i
        cons.append(1 - yi - ri)  # y_i <= 1 - r_i
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = pts[i]
            xj, yj, rj = pts[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            cons.append(dist_sq - sum_r**2)
    
    return obj, cons

def _solve_from_initial(x0, n, maxiter=1000, tol=1e-10):
    """
    Run SLSQP optimization from initial guess x0.
    Returns (result, success, obj_value)
    """
    # Convert constraints to SLSQP format
    # SLSQP expects: {'type': 'ineq', 'fun': lambda x: ...}
    # But we have many constraints. It's more efficient to define them separately or in a batch.
    # However, for 26 circles, number of non-overlap constraints is 26*25/2 = 325.
    # Plus 5*26 = 130 linear constraints. Total ~455 constraints.
    # SLSQP can handle this but it might be slow.
    
    # Let's define the constraint functions properly.
    
    def linear_constraints(x):
        n = x.shape[0] // 3
        pts = x.reshape(n, 3)
        c = np.zeros(5 * n)
        for i in range(n):
            xi, yi, ri = pts[i]
            c[5*i] = ri
            c[5*i+1] = xi - ri
            c[5*i+2] = 1 - xi - ri
            c[5*i+3] = yi - ri
            c[5*i+4] = 1 - yi - ri
        return c
    
    def overlap_constraints(x):
        n = x.shape[0] // 3
        pts = x.reshape(n, 3)
        m = n*(n-1)//2
        c = np.zeros(m)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = pts[i]
                xj, yj, rj = pts[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                c[idx] = dist_sq - sum_r**2
                idx += 1
        return c
    
    constraints = [
        {'type': 'ineq', 'fun': linear_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    bounds = [(0, 1) for _ in range(2*n)] + [(0, 1) for _ in range(n)]
    
    try:
        res = minimize(
            _obj_func, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': tol}
        )
        return res, res.success, -res.fun  # -res.fun is sum of radii
    except Exception as e:
        return None, False, 0.0

def _obj_func(x):
    """Objective function for minimize: -sum(radii)"""
    n = x.shape[0] // 3
    pts = x.reshape(n, 3)
    return -np.sum(pts[:, 2])

def _validate_and_fix(x, n):
    """
    Validate that the solution is feasible.
    If not, try to fix it by reducing radii slightly or adjusting positions.
    Returns a valid configuration or None.
    """
    pts = x.reshape(n, 3).copy()
    
    # Clip to boundaries
    for i in range(n):
        xi, yi, ri = pts[i]
        # Ensure ri >= 0
        ri = max(0.0, ri)
        # Ensure within boundaries
        xi = max(ri, min(1.0 - ri, xi))
        yi = max(ri, min(1.0 - ri, yi))
        pts[i] = [xi, yi, ri]
    
    # Check for overlaps and fix by reducing radii
    for _ in range(100):  # Iterate to fix overlaps
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = pts[i]
                xj, yj, rj = pts[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - 1e-9:
                    # Overlap exists. Reduce both radii.
                    # New radii should satisfy ri' + rj' <= dist
                    # Let's reduce proportionally
                    excess = (ri + rj) - dist
                    # Reduce each by half the excess, but not below 0
                    ri_new = max(0.0, ri - excess/2)
                    rj_new = max(0.0, rj - excess/2)
                    if ri_new + rj_new > dist + 1e-12:
                        # Still overlapping, reduce more
                        scale = dist / (ri + rj) if ri + rj > 0 else 1
                        ri_new = ri * scale * 0.999
                        rj_new = rj * scale * 0.999
                    pts[i, 2] = ri_new
                    pts[j, 2] = rj_new
                    fixed = False
        if fixed:
            break
    
    # Final validation
    for i in range(n):
        xi, yi, ri = pts[i]
        if xi < ri - 1e-7 or xi > 1 - ri + 1e-7 or yi < ri - 1e-7 or yi > 1 - ri + 1e-7:
            # Out of bounds
            ri = min(ri, xi, 1-xi, yi, 1-yi)
            ri = max(0.0, ri)
            pts[i, 2] = ri
    
    # Check overlaps again
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = pts[i]
            xj, yj, rj = pts[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-7:
                # Still overlapping, force reduce
                scale = max(0, dist) / (ri + rj) if ri + rj > 0 else 1
                pts[i, 2] *= scale * 0.999
                pts[j, 2] *= scale * 0.999
    
    return pts

def _generate_initial_guesses(n, num_guesses=20):
    """
    Generate multiple initial guesses for the optimization.
    Returns a list of initial guesses (each is an array of length 3n).
    """
    guesses = []
    
    # 1. Grid-based initial guess
    side = int(np.ceil(np.sqrt(n)))
    for i in range(side):
        for j in range(side):
            pass
    
    # Simple grid
    grid_side = int(np.ceil(np.sqrt(n)))
    cell_size = 1.0 / grid_side
    grid_guess = []
    count = 0
    for i in range(grid_side):
        for j in range(grid_side):
            if count >= n:
                break
            x = (i + 0.5) * cell_size
            y = (j + 0.5) * cell_size
            r = 0.4 * cell_size
            grid_guess.extend([x, y, r])
            count += 1
    guesses.append(np.array(grid_guess[:3*n]))
    
    # 2. Random initial guesses
    np.random.seed(42)
    for _ in range(num_guesses - 1):
        x_coords = np.random.uniform(0.1, 0.9, n)
        y_coords = np.random.uniform(0.1, 0.9, n)
        r_coords = np.random.uniform(0.02, 0.15, n)
        guess = np.concatenate([x_coords, y_coords, r_coords])
        guesses.append(guess)
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = N
    best_obj = -np.inf
    best_x = None
    
    # Generate initial guesses
    guesses = _generate_initial_guesses(n, num_guesses=15)
    
    # Try each initial guess
    for idx, x0 in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Run optimization
        res, success, obj_val = _solve_from_initial(x0, n, maxiter=500, tol=1e-9)
        
        if success and obj_val > best_obj:
            best_obj = obj_val
            best_x = x0 if res is None else res.x
        
        # Even if not successful, check if the result is valid
        if res is not None:
            x_candidate = res.x
            pts = _validate_and_fix(x_candidate, n)
            if pts is not None:
                sum_r = np.sum(pts[:, 2])
                if sum_r > best_obj:
                    best_obj = sum_r
                    best_x = pts.flatten()
    
    # If we haven't found a good solution, use the grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) for i in range(side) for j in range(side)]
        best_x = np.array(cells[:n]).flatten()
    
    # Final validation and polishing
    pts = _validate_and_fix(best_x, n)
    
    # One more optimization pass from the best solution
    if time.time() < deadline - 20:
        x0 = pts.flatten()
        res, success, obj_val = _solve_from_initial(x0, n, maxiter=1000, tol=1e-10)
        if success:
            pts_new = _validate_and_fix(res.x, n)
            if pts_new is not None and np.sum(pts_new[:, 2]) > np.sum(pts[:, 2]):
                pts = pts_new
    
    return pts
# EVOLVE-BLOCK-END
