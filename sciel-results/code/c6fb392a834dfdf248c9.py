"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _make_initial_grid(n_circles):
    """Create an initial grid placement in a 1x1 square."""
    # Use a grid layout
    cols = int(np.ceil(np.sqrt(n_circles)))
    rows = int(np.ceil(n_circles / cols))
    centers = []
    for i in range(n_circles):
        c = i % cols
        r = i // cols
        x = (c + 0.5) / cols
        y = (r + 0.5) / rows
        centers.append((x, y))
    return np.array(centers)

def _compute_objective(x, n_circles):
    """
    x = [w, h, x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    w + h = 2 (perimeter constraint, we fix this)
    Actually, let's parameterize differently.
    
    Let's use: x = [aspect, x1, y1, r1, ..., xn, yn, rn]
    where aspect = w/h, and w + h = 2.
    So w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    """
    aspect = x[0]
    if aspect <= 0.01:
        aspect = 0.01
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Constraints (converted to inequality form: g(x) >= 0)
    constraints = []
    
    # Wall constraints: circle must be inside rectangle
    for i in range(n_circles):
        # x_i - r_i >= 0
        # x_i + r_i <= w  =>  w - x_i - r_i >= 0
        # y_i - r_i >= 0
        # y_i + r_i <= h  =>  h - y_i - r_i >= 0
        constraints.append(lambda xi, wi, ri, i=i: xi - ri)
        constraints.append(lambda xi, wi, ri, w=w, i=i: w - xi - ri)
        constraints.append(lambda yi, hi, ri, i=i: yi - ri)
        constraints.append(lambda yi, hi, ri, h=h, i=i: h - yi - ri)
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            constraints.append(lambda xi, yi, ri, xj, yj, rj, i=i, j=j: 
                               (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2)
    
    return obj, constraints

def _objective_and_constraints(x, n_circles):
    """
    Combined objective and constraints for scipy.
    Returns (objective_value, constraint_values)
    """
    aspect = x[0]
    if aspect <= 0.01:
        aspect = 0.01
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    
    # Objective: minimize negative sum of radii
    obj = -np.sum(r)
    
    # Constraints in form g(x) >= 0
    con_vals = []
    
    for i in range(n_circles):
        con_vals.append(cx[i] - r[i])
        con_vals.append(w - cx[i] - r[i])
        con_vals.append(cy[i] - r[i])
        con_vals.append(h - cy[i] - r[i])
    
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            con_vals.append(dx*dx + dy*dy - (r[i]+r[j])**2)
    
    return obj, con_vals

def _is_valid(x, n_circles, tol=1e-6):
    """Check if a solution is valid."""
    aspect = x[0]
    if aspect <= 0.01:
        return False
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    
    if np.any(r < 0):
        return False
    
    # Check wall constraints
    for i in range(n_circles):
        if cx[i] - r[i] < -tol:
            return False
        if w - cx[i] - r[i] < -tol:
            return False
        if cy[i] - r[i] < -tol:
            return False
        if h - cy[i] - r[i] < -tol:
            return False
    
    # Check circle-circle constraints
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r[i] + r[j])**2
            if dist_sq < min_dist_sq - tol:
                return False
    
    return True

def _force_refine(x, n_circles, iterations=500, dt=0.001):
    """Physics-based force-directed refinement."""
    aspect = x[0]
    if aspect <= 0.01:
        aspect = 0.01
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    cx = x[1::3].copy()
    cy = x[2::3].copy()
    r = x[3::3].copy()
    
    for it in range(iterations):
        fx = np.zeros(n_circles)
        fy = np.zeros(n_circles)
        
        # Repulsion between circles
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx = 0.01 * (np.random.rand() - 0.5)
                    dy = 0.01 * (np.random.rand() - 0.5)
                    dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Strong repulsion
                    force = 10.0 * (min_dist - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Wall repulsion
        for i in range(n_circles):
            # Left wall
            if cx[i] < r[i]:
                fx[i] += 10.0 * (r[i] - cx[i])
            # Right wall
            if cx[i] + r[i] > w:
                fx[i] -= 10.0 * (cx[i] + r[i] - w)
            # Bottom wall
            if cy[i] < r[i]:
                fy[i] += 10.0 * (r[i] - cy[i])
            # Top wall
            if cy[i] + r[i] > h:
                fy[i] -= 10.0 * (cy[i] + r[i] - h)
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to valid region
        for i in range(n_circles):
            cx[i] = np.clip(cx[i], r[i], w - r[i])
            cy[i] = np.clip(cy[i], r[i], h - r[i])
    
    x_new = np.empty(1 + 3*n_circles)
    x_new[0] = aspect
    x_new[1::3] = cx
    x_new[2::3] = cy
    x_new[3::3] = r
    return x_new

def _solve_single(aspect_init, n_circles, deadline):
    """Solve the packing problem for a given initial aspect ratio."""
    # Initialize with grid
    centers = _make_initial_grid(n_circles)
    
    aspect = aspect_init
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    # Scale centers to fit in rectangle
    cx = centers[:, 0] * w
    cy = centers[:, 1] * h
    
    # Initial radius: small enough to fit
    min_spacing_x = w / int(np.ceil(np.sqrt(n_circles)))
    min_spacing_y = h / int(np.ceil(n_circles / int(np.ceil(np.sqrt(n_circles)))))
    r_init = 0.4 * min(min_spacing_x, min_spacing_y)
    r = np.full(n_circles, r_init)
    
    x = np.empty(1 + 3*n_circles)
    x[0] = aspect
    x[1::3] = cx
    x[2::3] = cy
    x[3::3] = r
    
    best_x = x.copy()
    best_obj = -np.sum(r)
    
    # Try to optimize
    try:
        # Define objective function
        def obj_func(xv):
            if time.time() > deadline:
                return np.inf
            o, _ = _objective_and_constraints(xv, n_circles)
            return o
        
        # Define constraints
        def con_func(xv):
            if time.time() > deadline:
                return np.zeros(4*n_circles + n_circles*(n_circles-1)//2)
            _, con = _objective_and_constraints(xv, n_circles)
            return np.array(con)
        
        # Bounds
        lb = [0.01] + [0.0]*n_circles + [0.0]*n_circles + [1e-6]*n_circles
        ub = [100.0] + [2.0]*n_circles + [2.0]*n_circles + [1.0]*n_circles
        
        res = minimize(
            obj_func,
            x,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=[{'type': 'ineq', 'fun': con_func}],
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if time.time() < deadline:
            if res.success or res.fun < best_obj:
                if _is_valid(res.x, n_circles):
                    best_x = res.x
                    best_obj = res.fun
    except Exception:
        pass
    
    return best_x, -best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    best_x = None
    best_sum_r = 0.0
    
    # Try multiple initial aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
        
        x, sum_r = _solve_single(aspect, N, deadline)
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_x = x
        
        # Force-directed refinement
        if time.time() < deadline - 10.0:
            x_refined = _force_refine(x, N, iterations=200, dt=0.001)
            if _is_valid(x_refined, N):
                sum_r_refined = np.sum(x_refined[3::3])
                if sum_r_refined > best_sum_r:
                    best_sum_r = sum_r_refined
                    best_x = x_refined
    
    # Final polish with tighter tolerance
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            def obj_func(xv):
                o, _ = _objective_and_constraints(xv, N)
                return o
            
            def con_func(xv):
                _, con = _objective_and_constraints(xv, N)
                return np.array(con)
            
            lb = [0.01] + [0.0]*N + [0.0]*N + [1e-6]*N
            ub = [100.0] + [2.0]*N + [2.0]*N + [1.0]*N
            
            res = minimize(
                obj_func,
                best_x,
                method='SLSQP',
                bounds=list(zip(lb, ub)),
                constraints=[{'type': 'ineq', 'fun': con_func}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if _is_valid(res.x, N):
                sum_r = np.sum(res.x[3::3])
                if sum_r > best_sum_r:
                    best_x = res.x
                    best_sum_r = sum_r
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            c = i % side
            r = i // side
            x = (c + 0.5) / side
            y = (r + 0.5) / side
            rad = 0.5 / side - 1e-3
            cells.append((x, y, rad))
        return np.array(cells)
    
    # Extract circles
    aspect = best_x[0]
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    cx = best_x[1::3]
    cy = best_x[2::3]
    r = best_x[3::3]
    
    # Scale to actual rectangle
    result = np.column_stack([cx, cy, r])
    
    return result

# EVOLVE-BLOCK-END
